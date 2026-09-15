#!/usr/bin/env python
"""Converte o benchmark NT (promoter_all) para o contrato de CSV do projeto.

Roda no LOGIN NODE depois de scripts/prefetch_dataset.sh. O job do benchmark
tambem o chama com --only-if-changed, que nao importa nada alem da stdlib
quando os dados ja estao atuais. Le os parquet do snapshot local do HF com
pyarrow, sem rede e NUNCA com `datasets`, e grava em --out-dir:
  train.csv, dev.csv, test.csv  cabecalho `sequence,label`, virgula, rotulo inteiro
                                (o mesmo contrato do smoke test)
  data_meta.json                origem, parametros, contagens, balanco,
                                comprimentos, N, IUPAC, duplicatas e sha256

Regras:
  - .upper() em toda sequencia (no NT, um trecho minusculo vira um unico <unk>);
  - codigos IUPAC ambiguos (RYKMSWBDHV) viram N, e a contagem e registrada;
    qualquer outro caractere fora de ACGTN aborta;
  - o HF so tem train e test. O dev e --dev-frac do train oficial, ESTRATIFICADO
    por classe com seed fixa e feito sobre sequencias UNICAS (copias identicas
    ficam do mesmo lado, para duplicatas internas do train nao vazarem para o
    dev). Isto NAO e o protocolo de 10 folds do paper: o numero que sai daqui e
    indicativo, nao comparavel ponto a ponto;
  - SOBREPOSICAO NO SPLIT OFICIAL: o promoter_all@96d86d56 tem 21 sequencias
    (0,35% do test, todas classe 1, mesmo rotulo) presentes no train E no test
    oficiais. Politica decidida (--sobreposicao-oficial, default
    remover-do-train): tira-las do pool de train ANTES do corte do dev, mantendo
    o test oficial intacto. A lista vai para o data_meta.json. Com `abortar`,
    qualquer sobreposicao oficial aborta;
  - depois disso, qualquer sequencia presente em mais de um split aborta
    (vazamento). Duplicatas internas a cada split sao so reportadas, nunca
    removidas em silencio.

Uso: prepare_data.py [--dataset-id ID] [--config promoter_all] [--revision SHA]
                     [--out-dir DIR] [--dev-frac 0.10] [--seed 42]
                     [--sobreposicao-oficial {remover-do-train,abortar}] [--only-if-changed]
     Defaults: NT_BENCH_DATASET_ID, NT_BENCH_CONFIG, NT_BENCH_DATASET_REVISION e
     NT_BENCH_DATA_DIR (environments/config.sh).
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
from collections import Counter

META_NOME = "data_meta.json"
SPLITS = ("train", "dev", "test")
IUPAC_AMBIGUOS = "RYKMSWBDHV"
PARA_N = str.maketrans(IUPAC_AMBIGUOS, "N" * len(IUPAC_AMBIGUOS))
PERMITIDOS = set("ACGTN") | set(IUPAC_AMBIGUOS)
FORMATO = 1  # versao do formato de saida; incrementar invalida --only-if-changed
NOTA_PROTOCOLO = ("dev = fracao estratificada do train oficial, seed fixa, sobre sequencias unicas. "
                  "NAO e o protocolo de 10 folds do paper do Nucleotide Transformer: metricas "
                  "obtidas com estes splits sao indicativas, nao comparaveis ponto a ponto.")


def sha256_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def parametros(args):
    return {"dataset_id": args.dataset_id, "config": args.config, "revision": args.revision,
            "dev_frac": args.dev_frac, "seed": args.seed, "sobreposicao_oficial": args.sobreposicao_oficial,
            "formato": FORMATO}


def dados_atuais(out_dir, params):
    """True se data_meta.json tem os mesmos parametros e os CSV batem com os sha256 gravados."""
    try:
        with open(os.path.join(out_dir, META_NOME)) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    if meta.get("parametros") != params:
        return False
    for split in SPLITS:
        path = os.path.join(out_dir, f"{split}.csv")
        if not os.path.exists(path) or sha256_arquivo(path) != meta.get("sha256", {}).get(split):
            return False
    return True


def localizar_snapshot(args):
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(args.dataset_id, repo_type="dataset", revision=args.revision,
                                 allow_patterns=[f"{args.config}/*.parquet"], local_files_only=True)
    except Exception as e:  # noqa: BLE001 (LocalEntryNotFoundError e afins variam entre versoes)
        sys.exit(f"ERRO: snapshot de {args.dataset_id}@{args.revision[:12]} nao encontrado no cache local "
                 f"(HF_HOME={os.environ.get('HF_HOME')}). Rode scripts/prefetch_dataset.sh no login node, "
                 f"com o MESMO HF_HOME. Erro original: {type(e).__name__}: {e}")


def ler_parquet(snap, config, split):
    """-> (sequencias cruas, rotulos int, nomes ou None, sha256 do parquet, colunas)."""
    import pyarrow.parquet as pq

    path = os.path.join(snap, config, f"{split}.parquet")
    if not os.path.exists(path):
        sys.exit(f"ERRO: {path} nao existe; o split oficial '{split}' e obrigatorio")
    dados = pq.read_table(path).to_pydict()
    faltando = {"sequence", "label"} - set(dados)
    if faltando:
        sys.exit(f"ERRO: {path} sem a(s) coluna(s) {sorted(faltando)}; colunas: {sorted(dados)}")
    # O repo agrupa 18 tarefas; a coluna `task` garante que so ha linhas desta.
    if "task" in dados:
        tarefas = Counter(dados["task"])
        if set(tarefas) != {config}:
            sys.exit(f"ERRO: {path} tem linhas de outras tarefas: {dict(tarefas)}")
    seqs = dados["sequence"]
    if any(not isinstance(s, str) for s in seqs):
        sys.exit(f"ERRO: {path} tem sequencia vazia ou nao-string")
    rotulos = []
    for i, y in enumerate(dados["label"]):
        if isinstance(y, bool) or not isinstance(y, int):
            sys.exit(f"ERRO: {path}, linha {i}: rotulo {y!r} nao e inteiro")
        rotulos.append(int(y))
    return seqs, rotulos, dados.get("name"), sha256_arquivo(path), sorted(dados)


def normalizar(seqs, origem):
    """.strip().upper() e IUPAC ambiguo -> N. Retorna (seqs, flags de linha alterada por IUPAC)."""
    saida, flags = [], []
    invalidos = Counter()
    for s in seqs:
        u = s.strip().upper()
        fora = set(u) - PERMITIDOS
        if fora:
            invalidos.update(fora)
            continue
        t = u.translate(PARA_N)
        saida.append(t)
        flags.append(t != u)
    if invalidos:
        sys.exit(f"ERRO: {origem} tem caracteres fora de ACGTN/IUPAC: {dict(invalidos.most_common(10))}")
    return saida, flags


def tratar_sobreposicao_oficial(train, test, politica):
    """Sequencias do train oficial que tambem estao no test oficial.

    train/test = (seqs, rotulos, nomes, flags). Com `remover-do-train`, devolve o
    train sem essas linhas e o registro do que saiu; com `abortar`, aborta.
    """
    comum = set(train[0]) & set(test[0])
    registro = {"politica": politica, "n_seqs": len(comum), "n_linhas_removidas_do_train": 0, "sequencias": []}
    if not comum:
        return train, registro
    if politica == "abortar":
        sys.exit(f"ERRO: VAZAMENTO no split OFICIAL: {len(comum)} sequencia(s) no train e no test oficiais; "
                 "nada foi gravado. Use --sobreposicao-oficial remover-do-train para tira-las do train "
                 "(registrado no data_meta.json).")
    nomes_test = test[2] or [None] * len(test[0])
    for s, y, nome in zip(test[0], test[1], nomes_test):
        if s in comum:
            registro["sequencias"].append({"sequence": s, "label_test": y, "name_test": nome})
    manter = [i for i, s in enumerate(train[0]) if s not in comum]
    registro["n_linhas_removidas_do_train"] = len(train[0]) - len(manter)
    train = tuple(None if v is None else [v[i] for i in manter] for v in train)
    print(f"AVISO: {len(comum)} sequencia(s) do test oficial tambem estao no train oficial; "
          f"{registro['n_linhas_removidas_do_train']} linha(s) removida(s) do train antes do corte do dev "
          "(politica remover-do-train; lista em data_meta.json)")
    return train, registro


def cortar_dev(seqs, frac, seed, rotulos):
    """Indices (train, dev): dev estratificado por classe sobre sequencias unicas."""
    from sklearn.model_selection import train_test_split

    rotulo_da_seq = {}
    for s, y in zip(seqs, rotulos):
        rotulo_da_seq.setdefault(s, y)  # conflitos de rotulo sao reportados em duplicatas()
    unicas = list(rotulo_da_seq)
    _, dev_unicas = train_test_split(unicas, test_size=frac, random_state=seed,
                                     stratify=[rotulo_da_seq[s] for s in unicas])
    dev_unicas = set(dev_unicas)
    idx_train = [i for i, s in enumerate(seqs) if s not in dev_unicas]
    idx_dev = [i for i, s in enumerate(seqs) if s in dev_unicas]
    return idx_train, idx_dev


def duplicatas(seqs, rotulos):
    contagem = Counter(seqs)
    rotulos_por_seq = {}
    for s, y in zip(seqs, rotulos):
        if contagem[s] > 1:
            rotulos_por_seq.setdefault(s, set()).add(y)
    return {
        "seqs_com_copias": sum(1 for v in contagem.values() if v > 1),
        "linhas_excedentes": sum(v - 1 for v in contagem.values() if v > 1),
        "seqs_com_rotulo_conflitante": sum(1 for r in rotulos_por_seq.values() if len(r) > 1),
    }


def estatisticas(seqs, rotulos, flags_iupac):
    comps = [len(s) for s in seqs]
    classes = Counter(rotulos)
    return {
        "n": len(seqs),
        "por_classe": {str(k): classes[k] for k in sorted(classes)},
        "frac_classe_1": round(classes.get(1, 0) / len(seqs), 4) if seqs else None,
        "comprimento_bp": {"min": min(comps), "medio": round(sum(comps) / len(comps), 1), "max": max(comps)},
        "n_com_N": sum(1 for s in seqs if "N" in s),
        "n_normalizadas_iupac": sum(flags_iupac),
        "duplicatas_internas": duplicatas(seqs, rotulos),
    }


def checar_vazamento(splits):
    conjuntos = {nome: set(seqs) for nome, (seqs, _, _) in splits.items()}
    sobreposicao, problemas = {}, []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            comum = conjuntos[a] & conjuntos[b]
            sobreposicao[f"{a}&{b}"] = len(comum)
            if comum:
                exemplos = ", ".join(s[:40] + "..." for s in sorted(comum)[:3])
                problemas.append(f"{a} & {b}: {len(comum)} sequencia(s) em comum (ex.: {exemplos})")
    if problemas:
        sys.exit("ERRO: VAZAMENTO entre splits; nada foi gravado.\n  " + "\n  ".join(problemas) +
                 "\n  Se a sobreposicao for train & test, ela vem do split OFICIAL do benchmark: decidir "
                 "explicitamente o que fazer, nunca deduplicar em silencio.")
    return sobreposicao


def escrever_csv(path, seqs, rotulos):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "label"])
        w.writerows(zip(seqs, rotulos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", default=os.environ.get("NT_BENCH_DATASET_ID"))
    ap.add_argument("--config", default=os.environ.get("NT_BENCH_CONFIG"),
                    help="subdiretorio do repositorio do HF com train/test.parquet")
    ap.add_argument("--revision", default=os.environ.get("NT_BENCH_DATASET_REVISION"))
    ap.add_argument("--out-dir", default=os.environ.get("NT_BENCH_DATA_DIR"))
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sobreposicao-oficial", choices=("remover-do-train", "abortar"), default="remover-do-train",
                    help="o que fazer com sequencias presentes no train E no test oficiais (default: "
                         "remover do train antes do corte do dev; o test oficial fica intacto)")
    ap.add_argument("--only-if-changed", action="store_true",
                    help="nao regera se data_meta.json tiver os mesmos parametros e os CSV baterem com os sha256")
    args = ap.parse_args()

    faltando = [n for n in ("dataset_id", "config", "revision", "out_dir") if not getattr(args, n)]
    if faltando:
        sys.exit(f"ERRO: faltam {', '.join('--' + n.replace('_', '-') for n in faltando)}; "
                 "passe por CLI ou carregue environments/config.sh (source) e exporte as NT_BENCH_*")
    if not 0 < args.dev_frac < 1:
        sys.exit(f"ERRO: --dev-frac {args.dev_frac} fora de (0, 1)")

    params = parametros(args)
    if args.only_if_changed and dados_atuais(args.out_dir, params):
        print(f"dados atuais em {args.out_dir} (parametros e sha256 batem), nada a fazer")
        return

    snap = localizar_snapshot(args)
    print(f"snapshot: {snap}")
    oficiais, origem_parquet = {}, {}
    for split in ("train", "test"):
        seqs, rotulos, nomes, sha_parquet, colunas = ler_parquet(snap, args.config, split)
        seqs, flags = normalizar(seqs, f"{args.config}/{split}.parquet")
        oficiais[split] = (seqs, rotulos, nomes, flags)
        origem_parquet[split] = {"linhas": len(seqs), "sha256": sha_parquet, "colunas": colunas}
        print(f"{split} oficial: {len(seqs)} linhas, colunas {colunas}")

    train_pool, sobreposicao_oficial = tratar_sobreposicao_oficial(oficiais["train"], oficiais["test"],
                                                                   args.sobreposicao_oficial)
    seqs, rotulos, _, flags = train_pool
    idx_train, idx_dev = cortar_dev(seqs, args.dev_frac, args.seed, rotulos)
    splits = {
        "train": tuple([v[i] for i in idx_train] for v in (seqs, rotulos, flags)),
        "dev": tuple([v[i] for i in idx_dev] for v in (seqs, rotulos, flags)),
        "test": (oficiais["test"][0], oficiais["test"][1], oficiais["test"][3]),
    }
    sobreposicao = checar_vazamento(splits)

    os.makedirs(args.out_dir, exist_ok=True)
    stats, hashes = {}, {}
    for split in SPLITS:
        s, y, f = splits[split]
        path = os.path.join(args.out_dir, f"{split}.csv")
        escrever_csv(path, s, y)
        hashes[split] = sha256_arquivo(path)
        stats[split] = estatisticas(s, y, f)
        st = stats[split]
        print(f"{split:5s}: {st['n']:6d} seqs, classes {st['por_classe']} (frac 1 = {st['frac_classe_1']}), "
              f"{st['comprimento_bp']['min']}/{st['comprimento_bp']['medio']}/{st['comprimento_bp']['max']} bp "
              f"(min/medio/max), com N {st['n_com_N']}, IUPAC->N {st['n_normalizadas_iupac']}, "
              f"duplicatas {st['duplicatas_internas']} -> {path}")
    combinado = hashlib.sha256("\n".join(f"{s}:{hashes[s]}" for s in SPLITS).encode()).hexdigest()

    import huggingface_hub
    import pyarrow
    import sklearn

    meta = {
        "parametros": params,
        "origem": {"dataset_id": args.dataset_id, "config": args.config, "revision": args.revision,
                   "parquet": origem_parquet},
        "nota_protocolo": NOTA_PROTOCOLO,
        "splits": stats,
        "sobreposicao_oficial_train_test": sobreposicao_oficial,
        "sobreposicao_entre_splits": sobreposicao,
        "sha256": hashes,
        "sha256_combinado": combinado,
        "sha256_combinado_definicao": 'sha256 de "train:<sha>\\ndev:<sha>\\ntest:<sha>"',
        "sequence_id_no_test": "indice 0-based da linha de dados em test.csv",
        "versoes": {"python": platform.python_version(), "pyarrow": pyarrow.__version__,
                    "scikit-learn": sklearn.__version__, "huggingface_hub": huggingface_hub.__version__},
        "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
    }
    with open(os.path.join(args.out_dir, META_NOME), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"sobreposicao entre splits: {sobreposicao}")
    print(f"sha256 combinado: {combinado}")
    print(f"meta : {os.path.join(args.out_dir, META_NOME)}")
    print(f"AVISO: {NOTA_PROTOCOLO}")


if __name__ == "__main__":
    main()
