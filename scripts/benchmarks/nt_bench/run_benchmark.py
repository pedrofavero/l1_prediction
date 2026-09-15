#!/usr/bin/env python
"""Benchmark NT promoter_all: fine-tuning + avaliacao, com criterio de aprovacao em faixa.

Etapa 2 da homologacao: validar o pipeline contra um benchmark com numero
publicado antes de usa-lo com L1. Referencia: F1 ~0,968 para o NT-v2-500m
(tabulacao de terceiros, arXiv 2502.18538; tratar como +-). Com o 50m
espera-se um pouco abaixo.

Criterio (F1 macro no test):
  0.90 <= F1 <= 0.98  PASS  exit 0
  F1 < 0.90           FAIL  exit 1, com diagnostico ordenado
  F1 > 0.98           WARN  exit 2: suspeita de vazamento ou avaliacao no split
                            errado (um 50m nao deveria superar o 500m)
Erro de ambiente, dados ou carga do modelo tambem termina em FAIL (exit 1).

O loop vem de scripts/lib/nt2_train.py (o mesmo do smoke test). Imports: so
torch, transformers, sklearn, numpy, stdlib e lib.nt2_train, NUNCA `datasets`.
Os pesados ficam dentro das funcoes, para --help funcionar sem torch. Requer
PYTHONPATH="$REPO_ROOT/scripts" (o .sbatch exporta).

Entrada: --data-dir com train/dev/test.csv + data_meta.json de prepare_data.py.
Os sha256 dos CSV sao conferidos contra o data_meta.json antes do treino.
Saidas em <report-dir>/run_<SLURM_JOB_ID ou timestamp>/:
  benchmark_report.json  sempre gravado, mesmo em falha
  predicoes_test.csv     sequence_id,label,pred,prob; sequence_id = indice 0-based
                         da linha de dados em test.csv, prob = P(classe 1)
"""
import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback

try:
    from lib.nt2_train import (MAX_LEN_TOKENS, carregar_classificador, carregar_csv, codificar, fixar_seed,
                               metricas, pico_vram, predizer, treinar)
except ModuleNotFoundError as e:
    if e.name not in ("lib", "lib.nt2_train"):
        raise
    sys.exit('ERRO: modulo lib.nt2_train nao encontrado (PYTHONPATH sem <repo>/scripts). Os .sbatch ja\n'
             'exportam; em uso interativo, antes de rodar:\n'
             '    export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"\n'
             '(REPO_ROOT = raiz do repositorio; `source environments/config.sh` a define)')

MODEL_ID_DEFAULT = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REVISION_DEFAULT = "81b29e5786726d891dbf929404ef20adca5b36f1"
SPLITS = ("train", "dev", "test")
NUM_LABELS = 2
F1_MIN = 0.90
F1_MAX = 0.98
F1_REFERENCIA_500M = 0.968
LR_REFERENCIA = 3e-5
CODIGO_SAIDA = {"PASS": 0, "FAIL": 1, "WARN": 2}


class Falha(Exception):
    """Falha esperada, com mensagem legivel (sem traceback)."""


def sha256_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def r4(x):
    return round(float(x), 4)


def checar_ambiente(report):
    import platform

    import numpy
    import sklearn
    import torch
    import transformers

    report["versoes"] = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "nt2_python": os.environ.get("NT2_PYTHON"), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "transformers": transformers.__version__, "numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
    }
    print("versoes: " + ", ".join(f"{k} {v}" for k, v in report["versoes"].items()))
    esperado = report["versoes"]["nt2_python"]
    if esperado and os.path.realpath(esperado) != os.path.realpath(sys.executable):
        print(f"AVISO: sys.executable ({sys.executable}) difere de NT2_PYTHON ({esperado})",
              file=sys.stderr, flush=True)
    if not torch.cuda.is_available():
        raise Falha("torch.cuda.is_available() == False: nenhuma GPU visivel para o job. Na particao shared o "
                    "recurso e --gres=mps:<pct>, NAO --gres=gpu:; confira as diretivas #SBATCH.")
    props = torch.cuda.get_device_properties(0)
    report["gpu"] = {"nome": props.name, "vram_total_gb": round(props.total_memory / 2**30, 1),
                     "compute_capability": f"{props.major}.{props.minor}",
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    print(f"GPU: {props.name}, {report['gpu']['vram_total_gb']} GB (compartilhada na shared)")
    return torch.device("cuda:0")


def carregar_dados(args, report):
    meta_path = os.path.join(args.data_dir, "data_meta.json")
    if not os.path.exists(meta_path):
        raise Falha(f"{meta_path} nao existe; rode scripts/benchmarks/nt_bench/prepare_data.py no login node")
    with open(meta_path) as f:
        meta = json.load(f)
    report["dados"] = {"data_dir": args.data_dir, "origem": meta.get("origem"), "parametros": meta.get("parametros"),
                       "sha256": {}, "sha256_combinado": meta.get("sha256_combinado"), "contagem": {}}
    splits = {}
    for nome in SPLITS:
        path = os.path.join(args.data_dir, f"{nome}.csv")
        if not os.path.exists(path):
            raise Falha(f"{path} nao existe; rode prepare_data.py --out-dir {args.data_dir}")
        h = sha256_arquivo(path)
        report["dados"]["sha256"][nome] = h
        if h != meta.get("sha256", {}).get(nome):
            raise Falha(f"sha256 de {path} ({h[:12]}) nao bate com {meta_path}: o CSV mudou depois do "
                        "prepare_data.py. Regere os dados; nao avalie um CSV sem rastreabilidade.")
        seqs, y = carregar_csv(path)
        if not set(y) <= {0, 1}:
            raise Falha(f"rotulos de {nome} fora de {{0,1}}: {sorted(set(y))[:10]}")
        splits[nome] = (seqs, y)
        report["dados"]["contagem"][nome] = {"n": len(y), "classe_1": sum(y)}
    print(f"dados: {report['dados']['contagem']} (sha256 conferidos com data_meta.json)")
    return splits


def carregar_modelo(args, dev, report):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
        model = carregar_classificador(args.model_id, args.revision, NUM_LABELS, dev)
    except OSError as e:
        raise Falha("modelo nao carregou; sem rede no no de compute, ele tem que estar no cache. Rode "
                    "scripts/prefetch_model.sh no login node com o MESMO HF_HOME "
                    f"({os.environ.get('HF_HOME')}). Erro original: {type(e).__name__}: {e}") from e
    report["modelo"] = {"params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")}
    print(f"modelo: {args.model_id}@{args.revision[:12]}, {report['modelo']['params_M']}M params")
    return tok, model


def gravar_predicoes(path, y, preds, probs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "label", "pred", "prob"])
        for i, (yi, pi, prob) in enumerate(zip(y, preds, probs[:, 1])):
            w.writerow([i, yi, pi, f"{prob:.6f}"])


def diagnostico_fail(splits, report, perdas, dev_hist, args):
    train_s = splits["train"][0]
    fora = sum(1 for s in train_s if set(s) - set("ACGTN"))
    tokens = report.get("tokens", {}).get("train", {})
    f1s = [d["f1_macro"] for d in dev_hist]
    melhor = max(range(len(f1s)), key=f1s.__getitem__) + 1 if f1s else None
    caiu = "NaN (lr alto demais?)" if any(math.isnan(p) for p in perdas) else \
        ("caiu" if perdas and perdas[-1] < perdas[0] else "NAO caiu")
    return [
        f"1. .upper() aplicado? seqs do train fora de ACGTN: {fora}; seqs do train com <unk>: "
        f"{tokens.get('n_com_unk')} (esperado 0; <unk> = sinal destruido, ex.: minusculas)",
        f"2. rotulos inteiros 0/1 e balanceados? {report['dados']['contagem']}; predicoes no test: "
        f"{report.get('test', {}).get('pred_distribuicao')} (tudo numa classe = nao aprendeu)",
        f"3. lr adequado? lr max {args.lr} (referencia {LR_REFERENCIA} para fine-tune completo); "
        f"loss por epoca {[r4(p) for p in perdas]}: {caiu}",
        f"4. dev acompanhou o train? F1 dev por epoca {f1s}; melhor epoca {melhor} de {args.epochs}. "
        "Loss caindo com F1 dev caindo = overfit (menos epocas ou lr menor); os dois ruins = underfit ou lr",
        "5. VRAM compartilhada na shared: se o .err mostrar OOM, outro job ocupou a GPU; reenvie",
    ]


def executar(args, report, out_dir):
    import resource

    import numpy as np
    import torch

    dev = checar_ambiente(report)
    splits = carregar_dados(args, report)

    fixar_seed(args.seed)  # antes do load: a cabeca de classificacao usa o RNG do torch
    tok, model = carregar_modelo(args, dev, report)

    t0 = time.perf_counter()
    enc = {nome: codificar(tok, seqs, MAX_LEN_TOKENS) for nome, (seqs, _) in splits.items()}
    report["tokens"] = {"tempo_tokenizacao_s": round(time.perf_counter() - t0, 1)}
    for nome, (ids, mask) in enc.items():
        reais = mask.sum(1).float()
        report["tokens"][nome] = {"medio": round(float(reais.mean()), 1), "max": int(reais.max()),
                                  "n_com_unk": int((ids == tok.unk_token_id).any(1).sum())}
    print(f"tokens por sequencia (padding 'longest' por lote): { {n: report['tokens'][n] for n in SPLITS} }")

    train_y, dev_y, test_y = (splits[n][1] for n in SPLITS)
    n_train = len(train_y)
    print(f"treino: {args.epochs} epocas x {math.ceil(n_train / args.batch_size)} passos, batch {args.batch_size}, "
          f"lr max {args.lr}, weight decay {args.weight_decay}, AdamW + OneCycleLR + clip 1.0, device {dev}")

    torch.cuda.reset_peak_memory_stats()
    dev_hist = []
    tempo_dev = [0.0]

    def ao_fim_da_epoca(ep, loss, lr_atual):
        t = time.perf_counter()
        preds, _ = predizer(model, *enc["dev"], args.eval_batch_size)
        m = metricas(dev_y, preds)
        tempo_dev[0] += time.perf_counter() - t
        dev_hist.append({"epoca": ep, "loss_treino": r4(loss), "lr_fim": lr_atual,
                         **{k: r4(v) for k, v in m.items() if k != "matriz_confusao"},
                         "matriz_confusao": m["matriz_confusao"]})
        print(f"epoca {ep}/{args.epochs}: loss {loss:.4f} | dev acc {m['accuracy']:.3f} F1 {m['f1_macro']:.3f} "
              f"MCC {m['mcc']:.3f} P {m['precision_macro']:.3f} R {m['recall_macro']:.3f} | lr {lr_atual:.2e}",
              flush=True)

    perdas = []
    t_treino = time.perf_counter()
    try:
        perdas = treinar(model, *enc["train"], train_y, args.epochs, args.batch_size, args.lr,
                         args.weight_decay, args.seed, ao_fim_da_epoca)
    finally:
        report["treino"] = {"loss_por_epoca": [r4(p) for p in perdas], "dev_por_epoca": dev_hist}
    torch.cuda.synchronize()
    tempo_treino = time.perf_counter() - t_treino
    tempo_so_treino = tempo_treino - tempo_dev[0]
    report["treino"].update(
        tempo_treino_s=round(tempo_treino, 1), tempo_avaliacao_dev_s=round(tempo_dev[0], 1),
        throughput_treino_seq_s=round(n_train * args.epochs / tempo_so_treino, 1),
    )

    t_inf = time.perf_counter()
    preds, probs = predizer(model, *enc["test"], args.eval_batch_size)
    torch.cuda.synchronize()
    tempo_inf = time.perf_counter() - t_inf
    m = metricas(test_y, preds)
    pred_path = os.path.join(out_dir, "predicoes_test.csv")
    gravar_predicoes(pred_path, test_y, preds, probs)
    report["test"] = {
        **{k: r4(v) for k, v in m.items() if k != "matriz_confusao"},
        "matriz_confusao": m["matriz_confusao"], "matriz_confusao_eixos": "linhas = verdade, colunas = predicao",
        "n": len(test_y), "pred_distribuicao": {str(c): int(np.sum(np.array(preds) == c)) for c in range(NUM_LABELS)},
        "tempo_inferencia_s": round(tempo_inf, 1), "throughput_inferencia_seq_s": round(len(test_y) / tempo_inf, 1),
        "predicoes_csv": pred_path,
    }
    alocada, reservada = pico_vram()
    report["recursos"] = {"vram_pico_alocada_gb": round(alocada, 2), "vram_pico_reservada_gb": round(reservada, 2),
                          "rss_host_pico_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print(f"test: acc {m['accuracy']:.4f} | F1 macro {m['f1_macro']:.4f} | MCC {m['mcc']:.4f} | "
          f"P {m['precision_macro']:.4f} | R {m['recall_macro']:.4f}")
    print(f"matriz de confusao (linhas = verdade, colunas = predicao): {m['matriz_confusao']}")
    print(f"treino {tempo_treino:.0f} s ({report['treino']['throughput_treino_seq_s']} seq/s sem a avaliacao no dev), "
          f"VRAM pico {alocada:.2f} GB, predicoes em {pred_path}")

    f1 = m["f1_macro"]
    if f1 < F1_MIN:
        diag = diagnostico_fail(splits, report, perdas, dev_hist, args)
        report["diagnostico"] = diag
        return "FAIL", f"F1 macro {f1:.4f} < {F1_MIN}. Diagnostico:\n   " + "\n   ".join(diag)
    if f1 > F1_MAX:
        return "WARN", (
            f"F1 macro {f1:.4f} > {F1_MAX}: um 50m nao deveria superar o 500m (~{F1_REFERENCIA_500M} publicado). "
            "Suspeitar de (1) vazamento por quase-duplicatas entre train e test (prepare_data.py so barra "
            "duplicata EXATA); (2) avaliacao no split errado: o test avaliado foi "
            f"{os.path.join(args.data_dir, 'test.csv')} sha256 {report['dados']['sha256']['test'][:12]}, "
            "confira com data_meta.json; (3) rotulo trivialmente derivavel da sequencia. NAO usar este "
            "resultado como homologacao.")
    return "PASS", (f"F1 macro {f1:.4f} em [{F1_MIN}, {F1_MAX}] (referencia 500m ~{F1_REFERENCIA_500M} +-; "
                    "com o 50m, um pouco abaixo e o esperado)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=os.environ.get("NT2_MODEL_ID", MODEL_ID_DEFAULT))
    ap.add_argument("--revision", default=os.environ.get("NT2_MODEL_REVISION", REVISION_DEFAULT))
    ap.add_argument("--data-dir", default=os.environ.get("NT_BENCH_DATA_DIR"),
                    help="train/dev/test.csv + data_meta.json de prepare_data.py (default: $NT_BENCH_DATA_DIR)")
    ap.add_argument("--report-dir", default=os.environ.get("NT_BENCH_REPORT_DIR", "."),
                    help="cria run_<jobid|timestamp>/ aqui (default: $NT_BENCH_REPORT_DIR ou cwd)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=LR_REFERENCIA)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    args = ap.parse_args()
    if not args.data_dir:
        sys.exit("ERRO: informe --data-dir ou exporte NT_BENCH_DATA_DIR (source environments/config.sh)")

    job = os.environ.get("SLURM_JOB_ID")
    out_dir = os.path.join(args.report_dir, f"run_{job or time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    hostname = socket.gethostname()
    inicio = time.time()
    print(f"benchmark NT | {hostname} | SLURM_JOB_ID={job} | {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"modelo {args.model_id} @ {args.revision} | dados {args.data_dir} | saida {out_dir}")

    report = {
        "resultado": "FAIL", "detalhe": "", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": hostname, "slurm_job_id": job, "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "argv": sys.argv, "model_id": args.model_id, "revision": args.revision,
        "hiperparametros": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "weight_decay": args.weight_decay, "seed": args.seed,
                            "eval_batch_size": args.eval_batch_size, "otimizador": "AdamW",
                            "scheduler": "OneCycleLR pct_start 0.1", "clip_grad": 1.0,
                            "padding": "longest por lote", "max_len_tokens": MAX_LEN_TOKENS},
        "criterio": {"metrica": "f1_macro no test", "pass": [F1_MIN, F1_MAX],
                     "referencia_nt_v2_500m": F1_REFERENCIA_500M, "codigos_saida": CODIGO_SAIDA},
    }
    try:
        resultado, detalhe = executar(args, report, out_dir)
    except Falha as e:
        resultado, detalhe = "FAIL", str(e)
    except Exception as e:  # noqa: BLE001
        resultado, detalhe = "FAIL", f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    report.update(resultado=resultado, detalhe=detalhe, duracao_s=round(time.time() - inicio, 1))
    caminho = os.path.join(out_dir, "benchmark_report.json")
    with open(caminho, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[{resultado}] {detalhe}")
    print(f"\nRESULTADO: {resultado} em {report['duracao_s']} s. Relatorio: {caminho}")
    sys.exit(CODIGO_SAIDA[resultado])


if __name__ == "__main__":
    main()
