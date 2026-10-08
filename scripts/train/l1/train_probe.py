#!/usr/bin/env python
"""Baseline L1: classificador sobre embeddings congelados (CPU).

Le os .npy de extract_embeddings.py, padroniza com media/desvio do train e treina
uma regressao logistica com C escolhido pelo MCC do dev (limiar 0,5). Com --mlp,
tambem um MLP de 1 camada oculta, com o mesmo protocolo. Cada linha (fwd e rc) e
um exemplo. Avalia dev, dev_strict, test e test_strict com metrics.py: por linha
e por janela (media de fwd/rc), nos limiares 0,5 e de MCC maximo no dev.

Antes de tudo confere: sha256 dos CSVs vs data_meta.json; sha256 do CSV registrado
na extracao; sha256 de cada .npy; window_id/orientation do index linha a linha.

Saida em <out-root>/<model_tag>/<ident>/<YYYYmmdd_HHMMSS>/ (nada compactado):
  report.json                 parametros, hashes, commit, selecao, limiares, metricas
  predictions_<split>.csv     window_id,orientation,source,label,prob,max_id_train,max_id_train_local
  predictions_mlp_<split>.csv com --mlp
  probe_logreg.json           coef, intercept, media e desvio do scaler
  probe_mlp_*.npy             pesos do MLP (com --mlp)

Requer PYTHONPATH="$REPO_ROOT/scripts" (lib.l1_dados); metrics.py vem do proprio diretorio.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import metrics as M

try:
    from lib.l1_dados import CONJUNTOS, IntegridadeErro, ler_csv_l1, sha256_arquivo, verificar_integridade
except ModuleNotFoundError as e:
    if e.name not in ("lib", "lib.l1_dados"):
        raise
    sys.exit('ERRO: modulo lib.l1_dados nao encontrado (PYTHONPATH sem <repo>/scripts). Antes de rodar:\n'
             '    export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"')

AVALIADOS = ("dev", "dev_strict", "test", "test_strict")
COLUNAS_PRED = ("window_id", "orientation", "source", "label", "prob", "max_id_train", "max_id_train_local")
DESVIO_MIN = 1e-6


class Falha(Exception):
    """Falha esperada, com mensagem legivel."""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lista_float(s):
    return [float(x) for x in s.split(",") if x.strip()]


def carregar(data_dir, emb_dir):
    """Confere tudo e devolve ({split: (X float32, conjunto)}, hashes dos CSVs, meta dos embeddings)."""
    import numpy as np

    try:
        hashes = verificar_integridade(data_dir)
    except IntegridadeErro as e:
        raise Falha(f"INTEGRIDADE: {e}") from None
    log("integridade dos CSVs OK")
    meta_path = os.path.join(emb_dir, "meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        raise Falha(f"{meta_path} ilegivel ({e}); rode extract_embeddings.py") from None
    dados = {}
    for s in CONJUNTOS:
        ent = meta.get("splits", {}).get(s)
        if not ent:
            raise Falha(f"{meta_path} sem o split {s}; rode extract_embeddings.py")
        if ent.get("sha256_csv") != hashes[s]:
            raise Falha(f"{s}: embeddings extraidos de outro CSV (sha256 {str(ent.get('sha256_csv'))[:12]} != "
                        f"{hashes[s][:12]}); reextraia")
        npy = os.path.join(emb_dir, f"{s}.npy")
        if not os.path.exists(npy) or sha256_arquivo(npy) != ent.get("sha256_npy"):
            raise Falha(f"{npy}: ausente ou sha256 diferente do registrado na extracao (corrompido?); reextraia")
        idx_path = os.path.join(emb_dir, f"{s}.index.csv")
        if not os.path.exists(idx_path) or sha256_arquivo(idx_path) != ent.get("sha256_index"):
            raise Falha(f"{idx_path}: ausente ou sha256 diferente do registrado na extracao; reextraia")
        csv_d = ler_csv_l1(os.path.join(data_dir, f"{s}.csv"))
        idx_d = ler_csv_l1(idx_path)
        if (idx_d["window_id"] != csv_d["window_id"] or idx_d["orientation"] != csv_d["orientation"]
                or idx_d["label"] != csv_d["label"]):
            raise Falha(f"{s}: ordem/rotulos do index.csv nao batem com o CSV linha a linha; reextraia")
        X = np.load(npy).astype(np.float32)
        if X.shape[0] != len(csv_d["label"]):
            raise Falha(f"{s}: {X.shape[0]} embeddings para {len(csv_d['label'])} linhas")
        conj = {"window_id": np.array(csv_d["window_id"]), "orientation": np.array(csv_d["orientation"]),
                "label": np.array(csv_d["label"], dtype=int), "source": np.array(csv_d["source"]),
                "max_id_train": np.array(csv_d["max_id_train"], dtype=float),
                "max_id_train_local": np.array(csv_d["max_id_train_local"], dtype=float)}
        dados[s] = (X, conj)
        log(f"{s}: {X.shape[0]} linhas x {X.shape[1]} dims, {int(conj['label'].sum())} positivas")
    return dados, hashes, meta


def selecionar(nome, fabricas, Xtr, ytr, Xdev, ydev):
    """Treina cada candidato no train e escolhe pelo MCC do dev (limiar 0,5); empate -> o primeiro da grade."""
    grade, melhor = [], None
    for param, fabrica in fabricas:
        t0 = time.time()
        clf = fabrica().fit(Xtr, ytr)
        p = clf.predict_proba(Xdev)[:, 1]
        mcc = M.binarias(ydev, p, 0.5)["mcc"]
        grade.append({"param": param, "mcc_dev": mcc, "tempo_s": round(time.time() - t0, 1)})
        log(f"{nome} {param}: MCC dev {mcc} ({grade[-1]['tempo_s']} s)")
        if melhor is None or mcc > melhor[1]:
            melhor = (param, mcc, clf)
    return melhor[2], melhor[0], grade


def gravar_predicoes(path, conj, prob):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(COLUNAS_PRED)
        for i in range(len(prob)):
            mi, ml = conj["max_id_train"][i], conj["max_id_train_local"][i]
            wr.writerow([conj["window_id"][i], conj["orientation"][i], conj["source"][i], int(conj["label"][i]),
                         f"{prob[i]:.6f}", "" if mi != mi else f"{mi:.4f}", "" if ml != ml else f"{ml:.4f}"])


def avaliar_modelo(clf, dados, scaler_x, out_dir, sufixo):
    """Probabilidades, limiares do dev, metricas e predicoes de um classificador ja treinado."""
    probs = {s: clf.predict_proba(scaler_x[s])[:, 1] for s in ("train",) + AVALIADOS}
    limiares = M.limiares_do_dev(dados["dev"][1], probs["dev"])
    res = {"limiares_dev": {k: M.r4(v) for k, v in limiares.items()},
           "train_total": M.binarias(dados["train"][1]["label"], probs["train"], 0.5), "conjuntos": {}}
    for s in AVALIADOS:
        res["conjuntos"][s] = M.avaliar_completo(dados[s][1], probs[s], limiares)
        gravar_predicoes(os.path.join(out_dir, f"predictions{sufixo}_{s}.csv"), dados[s][1], probs[s])
    return res


def imprimir_resumo(nome, res):
    print(f"\n=== {nome}: limiares do dev {res['limiares_dev']} | train MCC {res['train_total']['mcc']} ===")
    print(f"{'conjunto':12s} {'nivel':7s} {'limiar':6s} {'F1':>7s} {'MCC':>7s} {'AUROC':>7s} "
          f"{'MCC l1':>7s} {'MCC retro':>9s}")
    fmt = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
    for s in AVALIADOS:
        for nivel in ("linha", "janela"):
            for lim in ("0.5", "dev"):
                a = res["conjuntos"][s][nivel][lim]
                l1 = (a["l1_vs_negativos"] or {}).get("mcc")
                rv = (a["retrovirus_vs_negativos"] or {}).get("mcc")
                print(f"{s:12s} {nivel:7s} {lim:6s} {fmt(a['total']['f1_macro']):>7s} {fmt(a['total']['mcc']):>7s} "
                      f"{fmt(a['total']['auroc']):>7s} {fmt(l1):>7s} {fmt(rv):>9s}")
    faixas = [n for n, _ in M.FAIXAS_ID]
    print(f"\nrecall l1 por faixa (janela, limiar 0,5): n/recall")
    print(f"{'conjunto':12s} {'metrica':19s} " + " ".join(f"{f:>14s}" for f in faixas))
    for s in ("test", "test_strict", "dev", "dev_strict"):
        for met in ("max_id_train", "max_id_train_local"):
            d = res["conjuntos"][s]["janela"]["0.5"][f"faixas_{met}"]
            print(f"{s:12s} {met:19s} " + " ".join(f"{d[f]['n']:>6d}/{fmt(d[f]['recall']):>7s}" for f in faixas))
    a = res["conjuntos"]
    print("\nespecificidade/recall por source (janela, limiar 0,5): " +
          "; ".join(f"{s}: " + ", ".join(f"{k} {list(v.values())[1]}" for k, v in a[s]["janela"]["0.5"]["por_source"].items())
                    for s in ("test", "test_strict")))
    mt = a["test"]["linha"]["0.5"]["total"]["mcc"]
    ms = a["test_strict"]["linha"]["0.5"]["total"]["mcc"]
    if mt is not None and ms is not None and ms >= mt:
        print(f"AVISO: MCC do test_strict ({ms}) >= test ({mt}); o esperado e menor (menos vazamento). Investigar.")
    print(flush=True)


def git_commit():
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        sujo = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True,
                                   check=True).stdout.strip())
        return {"commit": sha, "arvore_modificada": sujo}
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--ident", required=True)
    ap.add_argument("--data-dir", help="padrao: $L1_DATASET_DIR/<ident>")
    ap.add_argument("--emb-dir", help="padrao: $L1_EMB_DIR/<model_tag>/<ident>")
    ap.add_argument("--out-root", default=env("L1_PROBE_DIR"), help="padrao: $L1_PROBE_DIR")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-grid", type=lista_float, default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--mlp", action="store_true", help="tambem treina um MLP de 1 camada oculta")
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--mlp-alpha-grid", type=lista_float, default="1e-4,1e-3,1e-2")
    args = ap.parse_args(argv)

    data_dir = args.data_dir or os.path.join(env("L1_DATASET_DIR", ""), args.ident)
    emb_dir = args.emb_dir or os.path.join(env("L1_EMB_DIR", ""), args.model_tag, args.ident)
    if not args.out_root:
        sys.exit("ERRO: passe --out-root ou defina L1_PROBE_DIR (environments/config.sh)")
    try:
        executar(args, data_dir, emb_dir)
    except Falha as e:
        sys.exit(f"ERRO: {e}")


def executar(args, data_dir, emb_dir):
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    t_ini = time.time()
    dados, hashes, meta_emb = carregar(data_dir, emb_dir)
    Xtr = dados["train"][0]
    media = Xtr.mean(0)
    desvio = np.maximum(Xtr.std(0), DESVIO_MIN)
    X = {s: (dados[s][0] - media) / desvio for s in dados}
    ytr, ydev = dados["train"][1]["label"], dados["dev"][1]["label"]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, args.model_tag, args.ident, ts)
    if os.path.exists(out_dir):
        out_dir += f"_{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
    os.makedirs(out_dir)
    log(f"saida: {out_dir}")

    modelos = {}
    fab_lr = [(f"C={c:g}", lambda c=c: LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=args.seed))
              for c in args.c_grid]
    lr, param_lr, grade_lr = selecionar("logreg", fab_lr, X["train"], ytr, X["dev"], ydev)
    with open(os.path.join(out_dir, "probe_logreg.json"), "w") as f:
        json.dump({"param": param_lr, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                   "scaler_media": media.tolist(), "scaler_desvio": desvio.tolist()}, f)
    modelos["logreg"] = {"selecionado": param_lr, "grade": grade_lr, **avaliar_modelo(lr, dados, X, out_dir, "")}

    if args.mlp:
        fab_mlp = [(f"alpha={a:g}", lambda a=a: MLPClassifier(hidden_layer_sizes=(args.mlp_hidden,), alpha=a,
                                                              max_iter=200, random_state=args.seed))
                   for a in args.mlp_alpha_grid]
        mlp, param_mlp, grade_mlp = selecionar("mlp", fab_mlp, X["train"], ytr, X["dev"], ydev)
        for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            np.save(os.path.join(out_dir, f"probe_mlp_w{i}.npy"), w)
            np.save(os.path.join(out_dir, f"probe_mlp_b{i}.npy"), b)
        np.save(os.path.join(out_dir, "probe_mlp_scaler_media.npy"), media)
        np.save(os.path.join(out_dir, "probe_mlp_scaler_desvio.npy"), desvio)
        modelos["mlp"] = {"selecionado": param_mlp, "grade": grade_mlp, "hidden": args.mlp_hidden,
                          **avaliar_modelo(mlp, dados, X, out_dir, "_mlp")}

    sha_npy = {s: e["sha256_npy"] for s, e in meta_emb["splits"].items()}
    report = {"parametros": {k: v for k, v in vars(args).items()}, "data_dir": data_dir, "emb_dir": emb_dir,
              "sha256_csv": hashes, "sha256_npy": sha_npy,
              "embeddings": {k: meta_emb.get(k) for k in ("model_tag", "familia", "model_id", "revision",
                                                          "origem_pesos", "versao_adaptador", "dim", "camada",
                                                          "pooling", "ambiente", "git")},
              "protocolo": {"padronizacao": "media/desvio do train (desvio minimo 1e-6)",
                            "selecao": "MCC no dev, por linha, limiar 0,5; empate -> primeiro da grade",
                            "limiar_dev": "MCC maximo no dev, por nivel (linha/janela); aplicado igual aos demais",
                            "exemplos": "cada linha (fwd e rc) e um exemplo",
                            "janela": "media das probabilidades de fwd e rc do mesmo window_id"},
              "modelos": modelos, "git": git_commit(),
              "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                          "scikit-learn": sklearn.__version__},
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": platform.node(),
              "tempo_s": round(time.time() - t_ini, 1), "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for nome, res in modelos.items():
        imprimir_resumo(nome, res)
    log(f"report: {os.path.join(out_dir, 'report.json')} ({report['tempo_s']} s)")


if __name__ == "__main__":
    main()
