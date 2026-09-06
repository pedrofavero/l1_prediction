#!/usr/bin/env python
"""Gera um dataset sintetico de classificacao binaria para o smoke test.

Desenho: fundo aleatorio com GC ~45%. A classe 1 recebe dois motivos de 12 bp
em posicoes aleatorias nao sobrepostas; a classe 0 recebe permutacoes das
mesmas letras. A composicao de nucleotideos fica quase identica entre as
classes, entao o modelo precisa aprender o motivo, nao o GC.

Por que 12 bp e nao 6: o NT tokeniza em frames de 6 ancorados na posicao 0.
Um motivo de 6 bp em posicao aleatoria so fica alinhado ao grid quando
pos % 6 == 0 (1/6 das vezes); nas outras 5/6 ele e partido entre dois tokens
meio motivo, meio fundo aleatorio, que mudam a cada exemplo. Com 12 bp,
qualquer janela de 12 posicoes contem pelo menos um bloco de 6 completo e
alinhado ao grid, em qualquer fase, entao sempre existe um token derivado
exclusivamente do motivo. O teste continua agnostico de tokenizador (serve
para o BPE do DNABERT-2 depois).

Os 6-mers dos motivos da classe 1 e da classe 0 sao disjuntos por construcao;
isso e verificado na carga do modulo (AssertionError se alguem editar os
motivos e introduzir um 6-mer compartilhado).

Saida: train.csv, dev.csv e test.csv com cabecalho `sequence,label`, separador
virgula e rotulo inteiro (o formato que o finetune/train.py do DNABERT-2
espera), mais dataset_meta.json com motivos, seed, seq_len e tamanhos, usado
por --only-if-changed para regerar quando a configuracao mudar.

Uso: make_toy_dataset.py [--out-dir DIR] [--seed 42] [--seq-len 300]
                         [--n-train 800] [--n-dev 200] [--n-test 200]
                         [--only-if-changed]
     --out-dir default: variavel de ambiente SMOKE_DATA_DIR.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

# Classe 0 usa as mesmas letras da classe 1, permutadas. O par GC NAO pode ser
# um repeat de periodo 6 (ex.: GGGCGGGGGCGG vs GCGGGGGCGGGG): os dois teriam
# exatamente o mesmo conjunto de 6-mers (as seis rotacoes de GGGCGG) e a
# checagem abaixo falharia. Os motivos atuais tem 6-mers disjuntos e nenhum
# deles contem GGGGGG.
MOTIVOS = {
    1: ("TATAAATATAAA", "GGGCGGGCGGGG"),
    0: ("ATAATAATAATA", "GGGGGCCGGGGG"),
}
BASES = np.array(list("ATCG"))
PROB = np.array([0.275, 0.275, 0.225, 0.225])  # GC ~45%
MAX_TENTATIVAS = 100
META_NOME = "dataset_meta.json"


def _kmers(seq, k=6):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _checar_6mers_disjuntos():
    k1 = set().union(*(_kmers(m) for m in MOTIVOS[1]))
    k0 = set().union(*(_kmers(m) for m in MOTIVOS[0]))
    colisao = sorted(k1 & k0)
    assert not colisao, (
        f"6-mers compartilhados entre os motivos da classe 1 e da classe 0: {colisao}. "
        "Isso deixa a tarefa mais dificil de forma silenciosa; escolha motivos disjuntos."
    )


_checar_6mers_disjuntos()


def gerar_sequencia(rng, seq_len, label):
    seq = rng.choice(BASES, size=seq_len, p=PROB)
    ocupadas = []
    for motivo in MOTIVOS[label]:
        k = len(motivo)
        for _ in range(MAX_TENTATIVAS):
            pos = int(rng.integers(0, seq_len - k + 1))
            if all(pos + k <= ini or pos >= fim for ini, fim in ocupadas):
                break
        else:
            raise ValueError(
                f"nao foi possivel posicionar o motivo {motivo} em {MAX_TENTATIVAS} tentativas: "
                f"--seq-len {seq_len} e curto demais para os motivos "
                f"({sum(len(m) for m in MOTIVOS[label])} bp no total)"
            )
        seq[pos:pos + k] = list(motivo)
        ocupadas.append((pos, pos + k))
    return "".join(seq)


def gerar_split(rng, n, seq_len):
    labels = np.array([0, 1] * (n // 2) + [0] * (n % 2))
    rng.shuffle(labels)
    return [(gerar_sequencia(rng, seq_len, int(lab)), int(lab)) for lab in labels]


def gc_medio(rows, label):
    seqs = [s for s, lab in rows if lab == label]
    if not seqs:
        return float("nan")
    return sum((s.count("G") + s.count("C")) / len(s) for s in seqs) / len(seqs)


def vazamento(rows):
    """Fracao de sequencias da classe 0 que contem por acaso algum motivo da classe 1."""
    seqs0 = [s for s, lab in rows if lab == 0]
    if not seqs0:
        return float("nan")
    return sum(any(m in s for m in MOTIVOS[1]) for s in seqs0) / len(seqs0)


def escrever_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "label"])
        w.writerows(rows)


def meta_atual(args):
    return {
        "motivos": {str(k): list(v) for k, v in MOTIVOS.items()},
        "seed": args.seed,
        "seq_len": args.seq_len,
        "n": {"train": args.n_train, "dev": args.n_dev, "test": args.n_test},
    }


def dataset_esta_atual(out_dir, meta):
    caminho = os.path.join(out_dir, META_NOME)
    if not os.path.exists(caminho):
        return False
    try:
        with open(caminho) as f:
            gravado = json.load(f)
    except (OSError, ValueError):
        return False
    if gravado != meta:
        return False
    return all(os.path.exists(os.path.join(out_dir, f"{s}.csv")) for s in ("train", "dev", "test"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.environ.get("SMOKE_DATA_DIR"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq-len", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--only-if-changed", action="store_true",
                    help="nao regera se dataset_meta.json existir e bater com a configuracao atual")
    args = ap.parse_args()

    if not args.out_dir:
        sys.exit("ERRO: informe --out-dir ou exporte SMOKE_DATA_DIR.")
    total_motivos = max(sum(len(m) for m in ms) for ms in MOTIVOS.values())
    if args.seq_len < total_motivos:
        sys.exit(f"ERRO: --seq-len {args.seq_len} e curto demais para os motivos ({total_motivos} bp no total).")

    meta = meta_atual(args)
    if args.only_if_changed and dataset_esta_atual(args.out_dir, meta):
        print(f"dataset atual em {args.out_dir} (meta bate com a configuracao), nada a fazer")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.n_train), ("dev", args.n_dev), ("test", args.n_test)):
        rows = gerar_split(rng, n, args.seq_len)
        path = os.path.join(args.out_dir, f"{split}.csv")
        escrever_csv(path, rows)
        n1 = sum(lab for _, lab in rows)
        print(f"{split:5s}: {len(rows):4d} sequencias ({len(rows) - n1} classe 0, {n1} classe 1), "
              f"{args.seq_len} bp, GC classe0={gc_medio(rows, 0):.3f} classe1={gc_medio(rows, 1):.3f}, "
              f"vazamento classe0->motivos classe1={vazamento(rows):.4f} -> {path}")

    with open(os.path.join(args.out_dir, META_NOME), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta : {os.path.join(args.out_dir, META_NOME)}")


if __name__ == "__main__":
    main()
