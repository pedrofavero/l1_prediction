"""Leitura e verificacao de integridade dos datasets L1 (gerados por scripts/data_prep/l1).

Biblioteca compartilhada por scripts/train/l1/extract_embeddings.py e train_probe.py.
Acesso via PYTHONPATH, nunca via sys.path no codigo:
    export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"
    from lib.l1_dados import verificar_integridade, ler_csv_l1

Regras do modulo (as mesmas de nt2_train.py): so stdlib; sem argparse, print ou
sys.exit; decisao e relatorio ficam nos consumidores.
"""
import csv
import hashlib
import json
import os

CONJUNTOS = ("train", "dev", "dev_strict", "test", "test_strict")
COLUNAS_INDICE = ("window_id", "orientation", "label", "source", "max_id_train", "max_id_train_local")


class IntegridadeErro(Exception):
    """CSV ausente ou com sha256 diferente do data_meta.json."""


def sha256_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_integridade(data_dir, conjuntos=CONJUNTOS):
    """sha256 de cada <conjunto>.csv contra data_meta.json["sha256"] -> {conjunto: sha256}.

    IntegridadeErro com todos os problemas encontrados: o storage do CISIA ja
    corrompeu arquivos, e um CSV sem rastreabilidade invalida qualquer metrica.
    """
    meta_path = os.path.join(data_dir, "data_meta.json")
    try:
        with open(meta_path) as f:
            esperado = json.load(f).get("sha256", {})
    except (OSError, ValueError) as e:
        raise IntegridadeErro(f"{meta_path} ilegivel ({type(e).__name__}: {e})") from None
    hashes, problemas = {}, []
    for nome in conjuntos:
        path = os.path.join(data_dir, f"{nome}.csv")
        if not os.path.exists(path):
            problemas.append(f"{nome}.csv nao existe")
            continue
        hashes[nome] = sha256_arquivo(path)
        if hashes[nome] != esperado.get(nome):
            problemas.append(f"{nome}.csv: sha256 {hashes[nome][:12]} != data_meta.json {str(esperado.get(nome))[:12]}")
    if problemas:
        raise IntegridadeErro(f"{data_dir}: " + "; ".join(problemas) + ". O arquivo mudou depois de gerado "
                              "(corrompido no storage ou copia incompleta): copie de novo do Mac e confira o "
                              "sha256 antes de usar.")
    return hashes


def _id_ou_none(x):
    return float(x) if x != "" else None


def ler_csv_l1(path, com_sequencia=False):
    """CSV do contrato L1 -> dict de listas na ordem das linhas.

    Chaves: COLUNAS_INDICE (label int; max_id_train* float ou None quando vazio,
    como no train) e `sequence` (.strip().upper()) se com_sequencia.
    ValueError se faltar coluna ou se um rotulo nao for 0/1.
    """
    cols = COLUNAS_INDICE + (("sequence",) if com_sequencia else ())
    out = {c: [] for c in cols}
    with open(path, newline="") as f:
        leitor = csv.DictReader(f)
        faltando = set(cols) - set(leitor.fieldnames or [])
        if faltando:
            raise ValueError(f"{path}: faltam as colunas {sorted(faltando)}")
        for n, row in enumerate(leitor, start=2):
            if row["label"] not in ("0", "1"):
                raise ValueError(f"{path}, linha {n}: rotulo {row['label']!r} fora de 0/1")
            out["window_id"].append(row["window_id"])
            out["orientation"].append(row["orientation"])
            out["label"].append(int(row["label"]))
            out["source"].append(row["source"])
            out["max_id_train"].append(_id_ou_none(row["max_id_train"]))
            out["max_id_train_local"].append(_id_ou_none(row["max_id_train_local"]))
            if com_sequencia:
                out["sequence"].append(row["sequence"].strip().upper())
    return out
