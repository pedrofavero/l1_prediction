#!/usr/bin/env python
"""Smoke test de infraestrutura: NT v2 50m ponta a ponta no cluster CISIA.

O objetivo NAO e avaliar o modelo. E provar que o servidor roda um modelo de
fundacao genomica de ponta a ponta (GPU via MPS, carga offline, tokenizacao,
forward, fine-tuning curto) antes de investir no dataset real de L1.

Etapas, em ordem (C = critica: falha marca o job como FAIL e termina com exit 1):
  1. ambiente      (C)  versoes, CUDA, GPU, matmul real na GPU
  2. carga_modelo  (C)  carga OFFLINE com revision pinada
  3. tokenizador        vocab, ids especiais, split 6-mer, efeito do N, minusculas
                        (cada checagem vira PASS/WARN; nunca aborta)
  4. forward       (C)  shapes de hidden_states/logits, mean pooling com mascara
  5. fine_tuning   (C)  treino real na GPU; accuracy >= 0.85 no teste (pulado com --quick)
  6. recursos           pico de VRAM, RSS do host e throughput

Imports pesados (torch, transformers, numpy, sklearn) ficam DENTRO das funcoes de
etapa, para que --help funcione sem GPU e para que um erro de import apareca
como FAIL da etapa, nao como crash antes do relatorio.

Caminhos vem de variaveis de ambiente exportadas pelo sbatch (NT2_MODEL_ID,
NT2_MODEL_REVISION, SMOKE_DATA_DIR, SMOKE_REPORT_DIR), com override por CLI.
Grava <report-dir>/smoke_report.json ao final, sempre, mesmo em falha.
"""
import argparse
import csv
import json
import os
import random
import socket
import sys
import time
import traceback

MODEL_ID_DEFAULT = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REVISION_DEFAULT = "81b29e5786726d891dbf929404ef20adca5b36f1"

SEQ_TESTE = "ACGTGTACGTGCACGGACGACTAGTCAGCA"  # 30 bp = 5 6-mers exatos
ESPECIAIS = {"<unk>": 0, "<pad>": 1, "<mask>": 2, "<cls>": 3, "<eos>": 4, "<bos>": 5}
VOCAB_ESPERADO = 4107
HIDDEN_ESPERADO = 512
N_CAMADAS_ESPERADO = 12
MAX_LEN_TOKENS = 2048
LIMIAR_ACC = 0.85

CTX = {}  # objetos compartilhados entre etapas: device, tokenizer, modelos, medicoes


class FalhaEtapa(Exception):
    """Falha esperada de uma etapa, com mensagem legivel (sem traceback)."""


# ---------------------------------------------------------------------------
# Infra do relatorio
# ---------------------------------------------------------------------------
class Relatorio:
    def __init__(self):
        self.etapas = {}
        self.ordem = []
        self.falha_critica = False

    def registrar(self, nome, status, detalhe="", **extra):
        linha = f"[{status}] {nome}"
        if detalhe:
            linha += f": {detalhe}"
        print(linha, flush=True)
        self.etapas[nome] = {"status": status, "detalhe": detalhe, **extra}
        if nome not in self.ordem:
            self.ordem.append(nome)


def etapa(rel, nome, fn, critica=True):
    """Executa fn(res) e registra PASS/FAIL/WARN/SKIP.

    fn recebe um dict `res` para preencher com medicoes (preservado mesmo em
    falha) e devolve a string de detalhe. FalhaEtapa ou qualquer excecao viram
    FAIL (critica) ou WARN (nao critica). Depois de uma falha critica, as etapas
    seguintes sao puladas.
    """
    print(f"\n=== {nome} ===", flush=True)
    if rel.falha_critica:
        rel.registrar(nome, "SKIP", "etapa critica anterior falhou")
        return
    res = {}
    t0 = time.time()
    try:
        detalhe = fn(res)
        status = "PASS"
    except FalhaEtapa as e:
        status = "FAIL" if critica else "WARN"
        detalhe = str(e)
    except Exception as e:  # noqa: BLE001
        status = "FAIL" if critica else "WARN"
        detalhe = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        print(res["traceback"], file=sys.stderr, flush=True)
    res["tempo_s"] = round(time.time() - t0, 2)
    res["critica"] = critica
    if status == "FAIL":
        rel.falha_critica = True
    rel.registrar(nome, status, detalhe, **res)


def fixar_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. ambiente
# ---------------------------------------------------------------------------
def etapa_ambiente(res):
    import platform

    import numpy
    import torch
    import transformers

    res.update(
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=numpy.__version__,
        torch_cuda_build=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    print(f"python {res['python']} | torch {res['torch']} (cuda {res['torch_cuda_build']}) | "
          f"transformers {res['transformers']} | numpy {res['numpy']}")

    if not torch.cuda.is_available():
        raise FalhaEtapa(
            "torch.cuda.is_available() == False: nenhuma GPU visivel para o job. "
            "Na particao shared o recurso e --gres=mps:<pct>, NAO --gres=gpu:; confira as "
            "diretivas #SBATCH. Se torch.version.cuda for None, o torch foi instalado sem "
            "CUDA: reinstale com --index-url https://download.pytorch.org/whl/cu128."
        )

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    res.update(
        gpu=props.name,
        vram_total_gb=round(props.total_memory / 2**30, 1),
        compute_capability=f"{props.major}.{props.minor}",
        n_gpus_visiveis=torch.cuda.device_count(),
    )
    print(f"GPU: {props.name}, {res['vram_total_gb']} GB (na shared a VRAM e compartilhada "
          f"entre jobs), compute capability {res['compute_capability']}")

    # Matmul real na GPU (nao so is_available): prova que o contexto CUDA funciona sob MPS
    n = 4096
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if not torch.isfinite(c).all():
        raise FalhaEtapa("matmul na GPU produziu valores nao finitos")
    res["matmul_4096_ms"] = round(dt * 1000, 2)
    res["matmul_tflops_fp32"] = round(2 * n**3 / dt / 1e12, 1)
    del a, b, c
    CTX["device"] = dev
    return (f"{props.name}, cc {res['compute_capability']}, matmul 4096x4096 em "
            f"{res['matmul_4096_ms']} ms ({res['matmul_tflops_fp32']} TFLOPS fp32)")


# ---------------------------------------------------------------------------
# 2. carga_modelo
# ---------------------------------------------------------------------------
def etapa_carga_modelo(res, args):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    res.update(model_id=args.model_id, revision=args.revision,
               hf_home=os.environ.get("HF_HOME"), hf_hub_offline=os.environ.get("HF_HUB_OFFLINE"))
    t0 = time.time()
    try:
        # trust_remote_code: o auto_map aponta para esm_config.py / modeling_esm.py
        # do proprio repositorio do HF. AutoModel NAO esta no auto_map; usar
        # AutoModelForMaskedLM (embeddings) ou AutoModelForSequenceClassification.
        tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        chaves = ("offline", "connection", "localentrynotfound", "couldn't connect", "cannot find",
                  "not found in cache", "network", "resolve", "we couldn't", "max retries", "outgoing traffic")
        if any(k in msg.lower() for k in chaves):
            raise FalhaEtapa(
                "modelo nao encontrado no cache local e sem rede no no de compute. Rode "
                "scripts/prefetch_model.sh no login node antes, com o MESMO HF_HOME "
                f"({os.environ.get('HF_HOME')}). Erro original: {msg}"
            ) from e
        raise
    res["tempo_carga_s"] = round(time.time() - t0, 1)
    res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    res["hidden_size"] = getattr(model.config, "hidden_size", None)
    res["num_hidden_layers"] = getattr(model.config, "num_hidden_layers", None)
    res["model_max_length"] = tok.model_max_length
    model.to(CTX["device"]).eval()
    CTX["tokenizer"] = tok
    CTX["model_mlm"] = model
    return (f"{res['params_M']}M params, hidden {res['hidden_size']}, {res['num_hidden_layers']} camadas, "
            f"carregado em {res['tempo_carga_s']} s (offline={res['hf_hub_offline']})")


# ---------------------------------------------------------------------------
# 3. tokenizador (nao critica: cada checagem vira PASS ou WARN)
# ---------------------------------------------------------------------------
def etapa_tokenizador(res, rel):
    tok = CTX["tokenizer"]
    avisos = []

    def tokens(seq):
        return tok.convert_ids_to_tokens(tok(seq)["input_ids"])

    def checar(nome, ok, detalhe):
        rel.registrar(f"tokenizador/{nome}", "PASS" if ok else "WARN", detalhe)
        res[nome] = {"ok": bool(ok), "detalhe": detalhe}
        if not ok:
            avisos.append(nome)

    # vocab: 6 especiais + 4096 6-mers + A/T/C/G/N = 4107
    checar("vocab", len(tok) == VOCAB_ESPERADO, f"len(tokenizer)={len(tok)}, esperado {VOCAB_ESPERADO}")

    ids = {t: tok.convert_tokens_to_ids(t) for t in ESPECIAIS}
    checar("ids_especiais", ids == ESPECIAIS, f"{ids}")

    # split 6-mer deterministico: <cls> + 5 6-mers (o NT nao adiciona <eos>)
    toks = tokens(SEQ_TESTE)
    esperado = ["<cls>"] + [SEQ_TESTE[i:i + 6] for i in range(0, len(SEQ_TESTE), 6)]
    checar("split_6mer", toks == esperado, f"{SEQ_TESTE} -> {toks}")

    # N reancora o grid de 6-mers: vira token proprio e custa tokens extras
    seq_n = SEQ_TESTE[:8] + "N" + SEQ_TESTE[9:]
    toks_n = tokens(seq_n)
    checar("efeito_N", len(toks_n) > len(toks) and "N" in toks_n,
           f"{len(toks)} tokens sem N -> {len(toks_n)} tokens com N na posicao 8: {toks_n}")

    # Minusculas: FASTA soft-masked (UCSC/Ensembl) traz repeticoes em minusculas e
    # o tokenizador e case-sensitive. Um trecho minusculo contiguo vira UM <unk>.
    # Nao da erro: so destroi o sinal. Todo pre-processamento deve aplicar .upper().
    minus = "acgtgt" * 5
    ids_min = tok(minus)["input_ids"]
    ids_up = tok(minus.upper())["input_ids"]
    unk_min = tok.unk_token_id in ids_min
    unk_up = tok.unk_token_id in ids_up
    checar("minusculas", unk_min and not unk_up,
           f"minusculas -> {tok.convert_ids_to_tokens(ids_min)} (<unk> presente: {unk_min}); "
           f".upper() -> {len(ids_up)} tokens, <unk> presente: {unk_up}. "
           "Todo pre-processamento deve aplicar .upper()")

    if avisos:
        raise FalhaEtapa(f"{len(avisos)} checagem(ns) com WARN: {', '.join(avisos)}. "
                         "Nao bloqueia o teste, mas revise antes de usar em dado real")
    return "vocab, ids especiais, split 6-mer, efeito do N e minusculas conforme o esperado"


# ---------------------------------------------------------------------------
# 4. forward
# ---------------------------------------------------------------------------
def etapa_forward(res):
    import torch

    tok, model, dev = CTX["tokenizer"], CTX["model_mlm"], CTX["device"]
    rng = random.Random(0)
    seqs = ["".join(rng.choice("ACGT") for _ in range(300)),
            "".join(rng.choice("ACGT") for _ in range(250))]  # tamanhos diferentes: exercita o padding

    # padding="longest" + truncation: o snippet do model card usa padding="max_length"
    # com max_length=2048, o que inflaria 300 bp para 2048 tokens.
    enc = tok(seqs, padding="longest", truncation=True, max_length=MAX_LEN_TOKENS, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    B, L = enc["input_ids"].shape
    res.update(batch=B, L_real=L, L_se_max_length=MAX_LEN_TOKENS, fator_inflacao=round(MAX_LEN_TOKENS / L, 1))
    print(f"batch {B}, {L} tokens com padding='longest' (padding='max_length' daria "
          f"{MAX_LEN_TOKENS}, {res['fator_inflacao']}x mais)")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    res["n_hidden_states"] = len(hs)
    res["last_hidden_shape"] = list(hs[-1].shape)
    res["logits_shape"] = list(out.logits.shape)
    print(f"hidden_states: {len(hs)} tensores; ultimo {res['last_hidden_shape']}; logits {res['logits_shape']}")

    problemas = []
    if len(hs) != N_CAMADAS_ESPERADO + 1:
        problemas.append(f"esperava {N_CAMADAS_ESPERADO + 1} hidden_states, veio {len(hs)}")
    if list(hs[-1].shape) != [B, L, HIDDEN_ESPERADO]:
        problemas.append(f"last hidden esperado [{B}, {L}, {HIDDEN_ESPERADO}], veio {res['last_hidden_shape']}")
    if list(out.logits.shape) != [B, L, VOCAB_ESPERADO]:
        problemas.append(f"logits esperado [{B}, {L}, {VOCAB_ESPERADO}], veio {res['logits_shape']}")

    # Mean pooling com mascara de atencao (embedding por sequencia)
    mask = enc["attention_mask"].unsqueeze(-1).to(hs[-1].dtype)
    pooled = (hs[-1] * mask).sum(1) / mask.sum(1)
    res["pooled_shape"] = list(pooled.shape)
    if not torch.isfinite(pooled).all():
        problemas.append("mean pooling produziu valores nao finitos")
    if not torch.isfinite(out.logits).all():
        problemas.append("logits nao finitos")
    res["vram_pico_forward_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if problemas:
        raise FalhaEtapa("; ".join(problemas))
    return (f"{len(hs)} hidden_states, last {res['last_hidden_shape']}, logits {res['logits_shape']}, "
            f"pooled {res['pooled_shape']} finito, VRAM pico {res['vram_pico_forward_gb']} GB")


# ---------------------------------------------------------------------------
# 5. fine_tuning
# ---------------------------------------------------------------------------
def ler_csv(path):
    seqs, labels = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seqs.append(row["sequence"].strip().upper())  # .upper(): FASTA soft-masked
            labels.append(int(row["label"]))
    return seqs, labels


def etapa_fine_tuning(res, args):
    import math

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from transformers import AutoModelForSequenceClassification

    if not args.data_dir:
        raise FalhaEtapa("informe --data-dir ou exporte SMOKE_DATA_DIR (gerado por make_toy_dataset.py)")
    splits = {}
    for nome in ("train", "dev", "test"):
        path = os.path.join(args.data_dir, f"{nome}.csv")
        if not os.path.exists(path):
            raise FalhaEtapa(f"{path} nao existe; rode make_toy_dataset.py --out-dir {args.data_dir}")
        splits[nome] = ler_csv(path)
    for nome, (s, y) in splits.items():
        if not set(y) <= {0, 1}:
            raise FalhaEtapa(f"rotulos de {nome} fora de {{0,1}}: {sorted(set(y))[:10]}")
    contagem = {nome: {"n": len(y), "classe_1": sum(y)} for nome, (_, y) in splits.items()}
    res["dados"] = contagem
    print(f"dados: {contagem}")

    fixar_seed(args.seed)
    tok, dev = CTX["tokenizer"], CTX["device"]

    # Libera o modelo MLM da etapa anterior antes de carregar o classificador
    if "model_mlm" in CTX:
        del CTX["model_mlm"]
        torch.cuda.empty_cache()

    # Cabeca de classificacao nativa (pooling CLS). Sem gradient checkpointing:
    # o modeling vendorizado tem supports_gradient_checkpointing = False.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, revision=args.revision, trust_remote_code=True, num_labels=2
    ).to(dev)

    def encode(batch_seqs):
        enc = tok(batch_seqs, padding="longest", truncation=True, max_length=MAX_LEN_TOKENS, return_tensors="pt")
        return {k: v.to(dev) for k, v in enc.items()}

    @torch.no_grad()
    def predizer(seqs, bs=64):
        model.eval()
        preds = []
        for i in range(0, len(seqs), bs):
            logits = model(**encode(seqs[i:i + bs])).logits
            preds.extend(logits.argmax(-1).tolist())
        return preds

    train_s, train_y = splits["train"]
    dev_s, dev_y = splits["dev"]
    test_s, test_y = splits["test"]
    bs = args.batch_size
    passos_epoca = math.ceil(len(train_s) / bs)
    total_passos = passos_epoca * args.epochs

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_passos, pct_start=0.1)
    print(f"treino: {args.epochs} epocas x {passos_epoca} passos, batch {bs}, lr max {args.lr}, "
          f"AdamW + OneCycleLR + clip 1.0, device {next(model.parameters()).device}")

    torch.cuda.reset_peak_memory_stats()
    perdas, accs_dev = [], []
    n_treinadas = 0
    t_treino = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        idx = list(range(len(train_s)))
        random.shuffle(idx)
        soma, n_batches = 0.0, 0
        for i in range(0, len(idx), bs):
            lote = idx[i:i + bs]
            enc = encode([train_s[j] for j in lote])
            y = torch.tensor([train_y[j] for j in lote], device=dev)
            out = model(**enc, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            soma += out.loss.item()
            n_batches += 1
            n_treinadas += len(lote)
        perdas.append(round(soma / n_batches, 4))
        acc_dev = accuracy_score(dev_y, predizer(dev_s))
        accs_dev.append(round(acc_dev, 4))
        print(f"epoca {ep + 1}/{args.epochs}: loss {perdas[-1]:.4f}, acc dev {acc_dev:.3f}, "
              f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
    torch.cuda.synchronize()
    tempo_treino = time.perf_counter() - t_treino

    t_inf = time.perf_counter()
    pred_test = predizer(test_s)
    torch.cuda.synchronize()
    tempo_inf = time.perf_counter() - t_inf

    acc = accuracy_score(test_y, pred_test)
    f1 = f1_score(test_y, pred_test, average="macro")
    mcc = matthews_corrcoef(test_y, pred_test)
    res.update(
        epochs=args.epochs, batch_size=bs, lr=args.lr, loss_por_epoca=perdas, acc_dev_por_epoca=accs_dev,
        test_accuracy=round(float(acc), 4), test_f1_macro=round(float(f1), 4), test_mcc=round(float(mcc), 4),
        limiar_accuracy=LIMIAR_ACC, tempo_treino_s=round(tempo_treino, 1),
        throughput_treino_seq_s=round(n_treinadas / tempo_treino, 1),
        throughput_inferencia_seq_s=round(len(test_s) / tempo_inf, 1),
        vram_pico_treino_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        pred_test_distribuicao={"0": int(np.sum(np.array(pred_test) == 0)), "1": int(np.sum(np.array(pred_test) == 1))},
    )
    CTX["throughput"] = {"treino_seq_s": res["throughput_treino_seq_s"],
                         "inferencia_seq_s": res["throughput_inferencia_seq_s"]}
    print(f"teste: accuracy {acc:.3f}, F1 macro {f1:.3f}, MCC {mcc:.3f} "
          f"(treino {tempo_treino:.0f} s, {res['throughput_treino_seq_s']} seq/s)")

    if acc < LIMIAR_ACC:
        diag = [
            f"1. .upper() aplicado? {'sim' if all(s == s.upper() for s in train_s[:50]) else 'NAO'} "
            "(ler_csv aplica; confira o CSV se veio de outro lugar)",
            f"2. rotulos inteiros 0/1 e balanceados? {contagem}",
            f"3. lr sensato? lr max {args.lr} (referencia: 1e-4 para o 50m); predicoes no teste: "
            f"{res['pred_test_distribuicao']} (tudo numa classe = nao aprendeu)",
            f"4. loss caiu ao longo das epocas? {perdas}; acc dev por epoca {accs_dev}",
            f"5. modelo na GPU? {next(model.parameters()).device}",
            "6. VRAM compartilhada na shared: se o .err mostrar OOM, outro job ocupou a GPU; reenvie",
        ]
        raise FalhaEtapa(f"accuracy {acc:.3f} < {LIMIAR_ACC}. Diagnostico:\n   " + "\n   ".join(diag))
    return f"accuracy {acc:.3f} >= {LIMIAR_ACC}, F1 macro {f1:.3f}, MCC {mcc:.3f}"


# ---------------------------------------------------------------------------
# 6. recursos
# ---------------------------------------------------------------------------
def etapa_recursos(res):
    import math
    import resource

    import torch

    pico = torch.cuda.max_memory_allocated() / 2**30
    reservado = torch.cuda.max_memory_reserved() / 2**30
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB no Linux
    rss_gb = rss_kb / 2**20
    res.update(vram_pico_alocada_gb=round(pico, 2), vram_pico_reservada_gb=round(reservado, 2),
               rss_host_pico_gb=round(rss_gb, 2), throughput=CTX.get("throughput"))
    mem_sugerida = max(8, math.ceil(rss_gb * 2))
    res["sugestao_sbatch"] = {"--mem": f"{mem_sugerida}G",
                              "--gres": "mps:25 (fine-tuning de modelo pequeno) ou gpu:h100:1 na particao gpu"}
    print(f"VRAM pico: {pico:.2f} GB alocada / {reservado:.2f} GB reservada (H100 tem 80 GB, "
          f"compartilhados na shared); RSS host pico {rss_gb:.2f} GB")
    if CTX.get("throughput"):
        print(f"throughput: treino {CTX['throughput']['treino_seq_s']} seq/s, "
              f"inferencia {CTX['throughput']['inferencia_seq_s']} seq/s (com --gres=mps:14)")
    return (f"VRAM pico {pico:.2f} GB, RSS {rss_gb:.2f} GB; ponto de partida para o job real: "
            f"--mem={mem_sugerida}G, --gres=mps:25")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=os.environ.get("NT2_MODEL_ID", MODEL_ID_DEFAULT))
    ap.add_argument("--revision", default=os.environ.get("NT2_MODEL_REVISION", REVISION_DEFAULT))
    ap.add_argument("--data-dir", default=os.environ.get("SMOKE_DATA_DIR"),
                    help="diretorio com train/dev/test.csv (default: $SMOKE_DATA_DIR)")
    ap.add_argument("--report-dir", default=os.environ.get("SMOKE_REPORT_DIR", "."),
                    help="onde gravar smoke_report.json (default: $SMOKE_REPORT_DIR ou cwd)")
    ap.add_argument("--quick", action="store_true", help="pula o fine-tuning (ambiente + carga + forward)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inicio = time.time()
    hostname = socket.gethostname()
    print(f"smoke test NT v2 | {hostname} | SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')} | "
          f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | quick={args.quick}")
    print(f"modelo {args.model_id} @ {args.revision}")

    rel = Relatorio()
    etapa(rel, "ambiente", etapa_ambiente)
    etapa(rel, "carga_modelo", lambda res: etapa_carga_modelo(res, args))
    etapa(rel, "tokenizador", lambda res: etapa_tokenizador(res, rel), critica=False)
    etapa(rel, "forward", etapa_forward)
    if args.quick:
        print("\n=== fine_tuning ===")
        rel.registrar("fine_tuning", "SKIP", "--quick")
    else:
        etapa(rel, "fine_tuning", lambda res: etapa_fine_tuning(res, args))
    etapa(rel, "recursos", etapa_recursos, critica=False)

    resultado = "FAIL" if rel.falha_critica else "PASS"
    report = {
        "resultado": resultado,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": hostname,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "argv": sys.argv,
        "model_id": args.model_id,
        "revision": args.revision,
        "quick": args.quick,
        "duracao_s": round(time.time() - inicio, 1),
        "ordem": rel.ordem,
        "etapas": rel.etapas,
    }
    os.makedirs(args.report_dir, exist_ok=True)
    caminho = os.path.join(args.report_dir, "smoke_report.json")
    with open(caminho, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== resumo ===")
    for nome in rel.ordem:
        print(f"  [{rel.etapas[nome]['status']}] {nome}")
    print(f"\nRESULTADO: {resultado} em {report['duracao_s']} s. Relatorio: {caminho}")
    sys.exit(1 if rel.falha_critica else 0)


if __name__ == "__main__":
    main()
