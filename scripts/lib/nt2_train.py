"""Treino e avaliacao de classificadores NT v2 via Hugging Face, em PyTorch puro.

Biblioteca compartilhada por:
  - scripts/smoke_tests/nt2_infra/smoke_test.py  (homologacao de infraestrutura)
  - scripts/benchmarks/nt_bench/run_benchmark.py (homologacao contra benchmark)
  - o treino de L1 (a seguir)

Acesso via PYTHONPATH, nunca via sys.path no codigo:
    export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"
    from lib.nt2_train import treinar, predizer, ...

Regras do modulo:
  - imports pesados (torch, transformers, sklearn, numpy) DENTRO das funcoes,
    para que --help de qualquer consumidor funcione sem carregar torch;
  - NUNCA importar `datasets`;
  - sem argparse, print, PASS/FAIL ou sys.exit: decisao e relatorio ficam nos
    consumidores.
"""
import csv
import random

MAX_LEN_TOKENS = 2048  # contexto maximo do NT v2, em tokens
PCT_WARMUP = 0.1       # pct_start do OneCycleLR
CLIP_GRAD = 1.0


def fixar_seed(seed):
    """Semeia random, numpy, torch e torch.cuda.

    Chamar ANTES de carregar_classificador(): a cabeca de classificacao e
    inicializada com o RNG do torch no load, antes de treinar().
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def carregar_csv(path):
    """Le um CSV do contrato `sequence,label` -> (seqs, labels).

    Aplica .strip().upper(): FASTA soft-masked (UCSC/Ensembl) traz repeticoes em
    minusculas e o tokenizador do NT e case-sensitive; um trecho minusculo
    contiguo vira UM <unk>. Colunas extras sao toleradas e ignoradas.
    ValueError se faltar `sequence` ou `label`, ou se um rotulo nao for inteiro.
    """
    seqs, labels = [], []
    with open(path, newline="") as f:
        leitor = csv.DictReader(f)
        faltando = {"sequence", "label"} - set(leitor.fieldnames or [])
        if faltando:
            raise ValueError(f"{path}: cabecalho {leitor.fieldnames} sem a(s) coluna(s) {sorted(faltando)}; "
                             "o contrato e `sequence,label`")
        for linha, row in enumerate(leitor, start=2):
            try:
                labels.append(int(row["label"]))
            except (TypeError, ValueError):
                raise ValueError(f"{path}, linha {linha}: rotulo {row['label']!r} nao e inteiro") from None
            seqs.append(row["sequence"].strip().upper())
    return seqs, labels


def codificar(tok, seqs, max_len):
    """Tokeniza `seqs` -> (input_ids, attention_mask), tensores long na CPU.

    padding="longest" + truncation=True. NUNCA padding="max_length": a 300 bp
    isso infla ~51 tokens para 2048, 40x mais compute. treinar() e predizer()
    recortam cada lote ao maior comprimento real do lote, entao o padding
    efetivo continua sendo "longest" por lote (exige padding a direita).

    LIMITACAO CONHECIDA: tokeniza o conjunto inteiro de uma vez (ansioso) e o
    guarda na RAM do host, com padding ate a maior sequencia do conjunto. Para
    o promoter_all (~48k x 51 tokens) sao ~20 MB por tensor, irrelevante. Para
    o L1 (~6 kb, ~1000 tokens, comprimento variavel) passa de centenas de MB; o
    treino de L1 provavelmente vai precisar de um Dataset com tokenizacao sob
    demanda. Ver scripts/lib/README.md, "Limitacoes conhecidas".
    """
    if tok.padding_side != "right":
        raise ValueError(f"tokenizer com padding_side={tok.padding_side!r}; "
                         "treinar()/predizer() assumem padding a direita")
    enc = tok(list(seqs), padding="longest", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def carregar_classificador(model_id, revision, num_labels, device):
    """AutoModelForSequenceClassification (cabeca nativa, pooling CLS), ja no device.

    trust_remote_code: o auto_map aponta para esm_config.py / modeling_esm.py do
    proprio repositorio do HF. Sem gradient checkpointing: o EsmModel
    vendorizado tem supports_gradient_checkpointing = False.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, num_labels=num_labels
    )
    return model.to(device)


def _lote(ids, mask, idx):
    """Linhas `idx` de ids/mask, sem as colunas que sao padding em todas elas."""
    m = mask[idx]
    comprimento = int(m.sum(1).max())
    return ids[idx][:, :comprimento], m[:, :comprimento]


def treinar(model, ids, mask, y, epochs, batch, lr, weight_decay, seed, ao_fim_da_epoca=None):
    """Fine-tuning completo: AdamW + OneCycleLR (pct_start 0.1) + clip de gradiente 1.0.

    ids/mask vem de codificar() (CPU); cada lote vai para o device do modelo.
    O embaralhamento usa um random.Random(seed) dedicado, que gera as mesmas
    permutacoes que o `random` global semeado com a mesma seed (o que o smoke
    test fazia antes da extracao deste loop). O RNG do torch NAO e re-semeado
    aqui: chame fixar_seed() antes de carregar_classificador().

    ao_fim_da_epoca(epoca, loss_media, lr_atual), com epoca 1-based, roda ao fim
    de cada epoca (ex.: avaliar no dev e imprimir). Pode chamar predizer(), que
    poe o modelo em eval; cada epoca recomeca com model.train().

    Retorna o historico de loss medio por epoca (lista de float).
    """
    import math

    import torch

    dev = next(model.parameters()).device
    y = torch.as_tensor(y, dtype=torch.long)
    n = len(y)
    passos_epoca = math.ceil(n / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos_epoca * epochs,
                                                pct_start=PCT_WARMUP)
    rng = random.Random(seed)
    historico = []
    for ep in range(epochs):
        model.train()
        idx = list(range(n))
        rng.shuffle(idx)
        soma, n_lotes = 0.0, 0
        for i in range(0, n, batch):
            lote = idx[i:i + batch]
            ids_b, mask_b = _lote(ids, mask, lote)
            out = model(input_ids=ids_b.to(dev), attention_mask=mask_b.to(dev), labels=y[lote].to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            soma += out.loss.item()
            n_lotes += 1
        historico.append(soma / n_lotes)
        if ao_fim_da_epoca is not None:
            ao_fim_da_epoca(ep + 1, historico[-1], sched.get_last_lr()[0])
    return historico


def predizer(model, ids, mask, batch):
    """Inferencia em lotes, sem gradiente -> (preds, probs).

    preds: list[int], argmax dos logits. probs: np.ndarray [N, num_labels],
    softmax em float32 (para 2 classes, probs[:, 1] = P(classe 1)).
    """
    import numpy as np
    import torch

    dev = next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            ids_b, mask_b = _lote(ids, mask, slice(i, i + batch))
            logits = model(input_ids=ids_b.to(dev), attention_mask=mask_b.to(dev)).logits
            preds.extend(logits.argmax(-1).tolist())
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, model.config.num_labels), dtype=np.float32)
    return preds, probs


def metricas(y_true, y_pred):
    """accuracy, f1/precision/recall macro, MCC e matriz de confusao (linhas = verdade, colunas = predicao)."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "matriz_confusao": confusion_matrix(y_true, y_pred).tolist(),
    }


def pico_vram():
    """(alocada_gb, reservada_gb): picos de VRAM do processo desde o ultimo reset_peak_memory_stats()."""
    import torch

    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
