#!/usr/bin/env python
"""Gera o dataset L1 (janelas de 1.024 bp) no contrato de CSV do projeto.

Roda no MAC, nao no cluster (environments/config.local.sh). So stdlib + pysam +
numpy; compativel com Python 3.10 e 3.13. Grava em --out-dir:
  train.csv, dev.csv, dev_strict.csv, test.csv, test_strict.csv
      sequence,label,source,detail,window_id,orientation,chrom,start,end,strand,group,gc,
      max_id_train,max_id_train_local
      (`sequence,label` primeiro: e o contrato de scripts/lib/nt2_train.carregar_csv)
  data_meta.json  parametros, sha256 das entradas e saidas, versoes, contagens,
                  GC, descartes, deficits, vazamento e checagens

Positivos (label 1): `l1` (janelas do hg38 com >= 200 bp cobertos pelo L1Farm com
Identity >= --min-identity) e `retrovirus` (RefSeq Retroviridae). Negativos
(label 0, 0 bp com L1/LTR/SVA): `te`, `background` e `markov`, casados em GC com o
`l1`. Toda janela entra nas duas orientacoes (fwd = fita de referencia, rc =
complemento reverso), sempre no mesmo arquivo.

Split sem vazamento: cromossomos de --holdout-chroms -> test; MMseqs2
(easy-linclust) sobre todas as janelas nas duas orientacoes; union-find de
cluster U elemento L1 U accession retroviral -> `group`. Dev: --dev-mode chrom
(padrao, cromossomos de --dev-chroms) ou hash (hash do grupo, --dev-frac).
dev_strict.csv = janelas de dev cujo grupo nao tem janela em train.
--leak-policy:
  filter-test (padrao) train intacto; test.csv = tudo do holdout;
                       test_strict.csv = so as janelas de test cujo grupo nao
                       tem nenhuma janela em train/dev (METRICA PRINCIPAL).
  purge-train          janelas fora do holdout de grupos que tocam o test sao
                       removidas; test_strict == test.

max_id_train / max_id_train_local: identidade maxima (MMseqs2 easy-search, duas
fitas) de cada janela de dev/test com qualquer janela do train; a primeira so
com hits de qcov >= 0.8, a segunda com alinhamentos >= 200 bp. Vazias no train.

Coordenadas: tudo 0-based half-open (BED), sem conversao: L1Farm e rmsk ja vem
assim (End - Start == LocusLength no L1Farm; isso e conferido na leitura).
"""
import argparse
import bisect
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict

PRIMARIOS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
FONTES = ("l1", "retrovirus", "te", "background", "markov")
FONTES_NEG = ("markov", "te", "background")
CONJUNTOS = ("train", "dev", "dev_strict", "test", "test_strict")
COLUNAS = ("sequence", "label", "source", "detail", "window_id", "orientation",
           "chrom", "start", "end", "strand", "group", "gc", "max_id_train", "max_id_train_local")
META_NOME = "data_meta.json"
FORMATO = 2  # versao do formato de saida; incrementar invalida --only-if-changed

MIN_COBERTURA_L1 = 200   # bp do L1Farm filtrado para a janela ser positiva; 1-199 = zona cinza
MAX_GAP_ELEMENTO = 500   # regioes do L1Farm a <= 500 bp (mesmo chrom e fita) = mesmo elemento
COBERTURA_MMSEQS = 0.8
# easy-linclust com mais de 1 thread muda a pertenca de algumas janelas entre runs (visto no dry run
# chr10/21/22: mesmo numero de grupos, membros diferentes) e quebra o determinismo dos CSVs.
THREADS_LINCLUST = 1
N_BINS_GC = 50           # bins de 2 pontos percentuais
OVERSAMPLE = 2.0         # pool de candidatos negativos = alvo estimado x OVERSAMPLE
OVERSAMPLE_TE_STRICT = 4.0  # pools de test/dev: Alu forma grupos entre cromossomos e muitos te saem do *_strict
MAX_LOTES_MARKOV = 30
LOTE_MARKOV_MAX = 50000
CLASSES_NAO_INTERCALADAS = {"Simple_repeat", "Low_complexity"}
PADRAO_POL = re.compile(r"\bpol\b|polymerase|reverse transcriptase", re.IGNORECASE)
COMPLEMENTO = str.maketrans("ACGT", "TGCA")
SEM_ACGT = str.maketrans("", "", "ACGT")
POLITICAS = ("filter-test", "purge-train")
MODOS_DEV = ("chrom", "hash")
# Busca de identidade com o train (easy-search): -c 0.19 ~ 200 bp de 1024, o minimo de L1 de um positivo
MIN_ID_BUSCA = 0.5
COBERTURA_BUSCA = 0.19
QCOV_MAX_ID = 0.8        # max_id_train: comparavel ao criterio do clustering (e ao *_strict)
ALNLEN_MAX_ID_LOCAL = 200  # max_id_train_local: trecho de L1 do tamanho minimo de um positivo
FAIXAS_ID = (("<0.80", 0.80), ("0.80-0.90", 0.90), ("0.90-0.95", 0.95), (">=0.95", None))
JUNTOS_PERMITIDOS = ({"train"}, {"dev", "dev_strict"}, {"test", "test_strict"})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def revcomp(s):
    return s.translate(COMPLEMENTO)[::-1]


def so_acgt(s):
    return not s.translate(SEM_ACGT)


def conta_gc(s):
    return s.count("G") + s.count("C")


def bin_gc(gc, comprimento):
    """gc em contagem de bases (inteiro): aritmetica inteira, sem erro de ponto flutuante na borda do bin."""
    return min(gc * N_BINS_GC // comprimento, N_BINS_GC - 1)


def semente(seed, *rotulo):
    """Seed derivada e estavel (nao depende de PYTHONHASHSEED) para cada etapa aleatoria."""
    chave = ":".join(str(x) for x in (seed,) + rotulo)
    return int(hashlib.sha256(chave.encode()).hexdigest()[:16], 16)


def fracao_hash(seed, chave):
    """Numero em [0, 1) deterministico a partir de (seed, chave)."""
    return int(hashlib.sha256(f"{seed}:{chave}".encode()).hexdigest()[:15], 16) / 16 ** 15


def por_fracoes(u, fracoes):
    """u em [0, 1) -> train/dev/test segundo fracoes (train, dev, test)."""
    if u < fracoes[0]:
        return "train"
    if u < fracoes[0] + fracoes[1]:
        return "dev"
    return "test"


def sha256_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def e_familia_l1(familia):
    """repFamily L1 do rmsk (inclui variantes como `L1-Tx1`): L1 antigo e L1, nunca negativo."""
    return familia.startswith("L1")


class Intervalos:
    """Uniao de intervalos 0-based half-open de um cromossomo, com consulta de sobreposicao em bp."""

    def __init__(self, pares=()):
        self.starts, self.ends = [], []
        for s, e in sorted(pares):
            if e <= s:
                continue
            if self.ends and s <= self.ends[-1]:
                if e > self.ends[-1]:
                    self.ends[-1] = e
            else:
                self.starts.append(s)
                self.ends.append(e)
        self.acum = [0]
        for s, e in zip(self.starts, self.ends):
            self.acum.append(self.acum[-1] + e - s)

    def sobreposicao(self, s, e):
        """bp de [s, e) cobertos pela uniao."""
        j = bisect.bisect_right(self.ends, s)
        k = bisect.bisect_left(self.starts, e)
        if j >= k:
            return 0
        total = self.acum[k] - self.acum[j]
        total -= max(0, s - self.starts[j])
        total -= max(0, self.ends[k - 1] - e)
        return total

    def lacunas(self, inicio, fim):
        """Trechos de [inicio, fim) fora da uniao."""
        pos = inicio
        for s, e in zip(self.starts, self.ends):
            if e <= pos:
                continue
            if s >= fim:
                break
            if s > pos:
                yield pos, s
            pos = e
        if pos < fim:
            yield pos, fim


# --------------------------------------------------------------------------- leitura

def ler_l1farm(path, cromossomos):
    """TSV do L1Farm (sem cabecalho; `#` e linhas em branco ignoradas) -> (regioes, n fora dos cromossomos).

    12 colunas: Chromosome, Start, End, Subfamily, Region, Strand, Mismatches,
    Deletions, Insertions, LocusLength, Identity, Similarity. 0-based half-open:
    aborta se End - Start != LocusLength (mudanca de convencao na fonte).
    """
    regioes, fora = [], 0
    with open(path) as f:
        for n, linha in enumerate(f, 1):
            linha = linha.rstrip("\r\n")
            if not linha.strip() or linha.startswith("#"):
                continue
            c = linha.split("\t")
            if len(c) != 12:
                raise ValueError(f"{path}:{n}: {len(c)} colunas, esperadas 12")
            if c[0] not in cromossomos:
                fora += 1
                continue
            s, e = int(c[1]), int(c[2])
            if e - s != int(c[9]):
                raise ValueError(f"{path}:{n}: End-Start={e - s} != LocusLength={c[9]}; "
                                 "a convencao 0-based half-open nao vale para este arquivo")
            regioes.append({"chrom": c[0], "start": s, "end": e, "subfamily": c[3], "region": c[4],
                            "strand": c[5], "identity": float(c[10])})
    return regioes, fora


def ler_rmsk(path, cromossomos, te_classes):
    """rmsk.txt.gz do UCSC -> {chrom: {"l1", "ltr_sva", "intercalado": [(s, e)], "te": [(s, e, strand, nome, classe, familia)]}}.

    Colunas 0-based: 5 genoName, 6 genoStart, 7 genoEnd, 9 strand, 10 repName,
    11 repClass, 12 repFamily. LTR/SVA para exclusao de negativos: repClass LTR
    (inclusive `LTR?`), repFamily SVA ou repClass Retroposon. Candidatos `te`:
    repClass em te_classes por igualdade exata (`DNA?` fica fora); LINE ja vem
    sem L1 porque L1 e testado antes.
    """
    out = {c: {"l1": [], "ltr_sva": [], "intercalado": [], "te": []} for c in cromossomos}
    with gzip.open(path, "rt") as f:
        for linha in f:
            c = linha.split("\t", 13)
            d = out.get(c[5])
            if d is None:
                continue
            s, e, classe, familia = int(c[6]), int(c[7]), c[11], c[12]
            if classe not in CLASSES_NAO_INTERCALADAS:
                d["intercalado"].append((s, e))
            if e_familia_l1(familia):
                d["l1"].append((s, e))
            elif classe.rstrip("?") == "LTR" or familia == "SVA" or classe == "Retroposon":
                d["ltr_sva"].append((s, e))
            elif classe in te_classes:
                d["te"].append((s, e, sys.intern(c[9]), sys.intern(c[10]), sys.intern(classe), sys.intern(familia)))
    return out


def agrupar_elementos(regioes, max_gap=MAX_GAP_ELEMENTO):
    """Regioes no mesmo (chrom, fita) a <= max_gap bp viram um elemento (componente conexo).

    element_id = chrom:fita:start-end do span do elemento.
    """
    por_chave = defaultdict(list)
    for r in regioes:
        por_chave[(r["chrom"], r["strand"])].append(r)
    elementos = []
    for chrom, strand in sorted(por_chave):
        atual = None
        for r in sorted(por_chave[(chrom, strand)], key=lambda r: (r["start"], r["end"])):
            if atual is not None and r["start"] - atual["end"] <= max_gap:
                atual["regions"].append(r)
                atual["end"] = max(atual["end"], r["end"])
            else:
                atual = {"chrom": chrom, "strand": strand, "start": r["start"], "end": r["end"], "regions": [r]}
                elementos.append(atual)
    for el in elementos:
        el["element_id"] = f"{el['chrom']}:{el['strand']}:{el['start']}-{el['end']}"
    return elementos


def starts_janela(inicio, fim, w, stride, limite=None):
    """Starts das janelas de w bp sobre [inicio, fim).

    Menor que w: uma janela centrada (encostada na borda 0/limite se preciso).
    Senao: passo `stride` a partir de inicio + uma janela final alinhada a fim.
    """
    if fim - inicio < w:
        s = (inicio + fim) // 2 - w // 2
        if limite is not None:
            s = min(s, limite - w)
        return [max(0, s)]
    starts = list(range(inicio, fim - w + 1, stride))
    if starts[-1] + w < fim:
        starts.append(fim - w)
    return starts


def nova_janela(wid, source, label, seq, **campos):
    j = {"wid": wid, "source": source, "label": label, "seq": seq, "gc": conta_gc(seq),
         "chrom": "", "start": "", "end": "", "strand": "", "detail": "", "uniao": [], "holdout": False,
         "dev_chrom": False,
         "subfamily": "", "rep_family": "", "accession": ""}
    j.update(campos)
    return j


def descrever_l1(regioes, s, e):
    """-> (detail `SUB:REG,REG;SUB:REG`, subfamilia principal = maior sobreposicao em bp)."""
    por_sub, bp = defaultdict(set), Counter()
    for r in regioes:
        ov = min(e, r["end"]) - max(s, r["start"])
        if ov > 0:
            por_sub[r["subfamily"]].add(r["region"])
            bp[r["subfamily"]] += ov
    detail = ";".join(f"{sf}:{','.join(sorted(por_sub[sf]))}" for sf in sorted(por_sub))
    principal = min(bp, key=lambda sf: (-bp[sf], sf)) if bp else ""
    return detail, principal


class RegioesOrdenadas:
    """Regioes L1Farm de um cromossomo, para listar as que tocam [s, e)."""

    def __init__(self, regioes):
        self.regioes = sorted(regioes, key=lambda r: (r["start"], r["end"]))
        self.starts = [r["start"] for r in self.regioes]
        self.max_len = max((r["end"] - r["start"] for r in self.regioes), default=0)

    def tocando(self, s, e):
        i = bisect.bisect_left(self.starts, s - self.max_len)
        k = bisect.bisect_left(self.starts, e)
        return [r for r in self.regioes[i:k] if r["end"] > s]


def extrair_l1(chrom, seq, elementos, regioes_filtradas, w, stride, holdout, cont):
    """Janelas `l1` de um cromossomo (seq ja em maiusculas). Dedup por coordenada.

    Uma janela gerada por dois elementos guarda os dois em `uniao` (mesmo grupo).
    Cobertura >= MIN_COBERTURA_L1 -> positivo; 1-199 -> zona cinza; com N/IUPAC -> descarte.
    """
    cobertura = Intervalos((r["start"], r["end"]) for r in regioes_filtradas)
    regs = RegioesOrdenadas(regioes_filtradas)
    por_start = {}
    for el in elementos:
        for s in starts_janela(el["start"], el["end"], w, stride, len(seq)):
            j = por_start.get(s)
            if j is None:
                por_start[s] = {"strand": el["strand"], "uniao": ["elem:" + el["element_id"]]}
            else:
                cont["l1_janela_em_2_elementos"] += 1
                j["uniao"].append("elem:" + el["element_id"])
    out = []
    for s in sorted(por_start):
        e = s + w
        cob = cobertura.sobreposicao(s, e)
        if cob < MIN_COBERTURA_L1:
            cont["l1_zona_cinza_1a199bp" if cob > 0 else "l1_sem_cobertura"] += 1
            continue
        sub = seq[s:e]
        if len(sub) != w:
            cont["l1_borda_cromossomo"] += 1
            continue
        if not so_acgt(sub):
            cont["l1_com_N" if "N" in sub else "l1_iupac"] += 1
            continue
        detail, principal = descrever_l1(regs.tocando(s, e), s, e)
        info = por_start[s]
        out.append(nova_janela(f"l1:{chrom}:{s}-{e}", "l1", 1, sub, chrom=chrom, start=s, end=e,
                               strand=info["strand"], detail=detail, uniao=info["uniao"],
                               holdout=holdout, subfamily=principal))
    return out


def janela_centrada(s0, e0, w, limite):
    s = (s0 + e0) // 2 - w // 2
    return max(0, min(s, limite - w))


def candidatos_te(seq, elementos_te, l1_any, excl, w, cont):
    """Janelas centradas nos elementos te (sem sequencia): [(start, strand, nome, classe, familia, gc)]."""
    vistos, out = set(), []
    for s0, e0, strand, nome, classe, familia in sorted(elementos_te):
        s = janela_centrada(s0, e0, w, len(seq))
        if s in vistos:
            cont["te_janela_duplicada"] += 1
            continue
        vistos.add(s)
        e = s + w
        if l1_any.sobreposicao(s, e):
            cont["neg_zona_cinza_toca_l1"] += 1
            continue
        if excl.sobreposicao(s, e):
            cont["te_toca_ltr_sva"] += 1
            continue
        sub = seq[s:e]
        if len(sub) != w:
            cont["te_borda_cromossomo"] += 1
            continue
        if not so_acgt(sub):
            cont["te_com_N" if "N" in sub else "te_iupac"] += 1
            continue
        out.append((s, strand, nome, classe, familia, conta_gc(sub)))
    return out


def candidatos_background(seq, bloqueio, w, cont):
    """Tiles nao sobrepostos de w bp nas lacunas do bloqueio: [(start, gc)]."""
    out = []
    for g0, g1 in bloqueio.lacunas(0, len(seq)):
        for s in range(g0, g1 - w + 1, w):
            sub = seq[s:s + w]
            if not so_acgt(sub):
                cont["background_com_N" if "N" in sub else "background_iupac"] += 1
                continue
            out.append((s, conta_gc(sub)))
    return out


def janelas_retrovirus(path_genomas, path_cds, modo, w, stride, cont):
    """Janelas `retrovirus`. genome: tiles sobre os genomas; pol: so CDS de pol/polimerase/RT."""
    import pysam

    out, produtos = [], Counter()
    with pysam.FastxFile(path_genomas if modo == "genome" else path_cds) as fx:
        for rec in fx:
            if modo == "pol":
                comentario = rec.comment or ""
                produto = comentario.split("|")[1] if comentario.count("|") >= 1 else comentario
                if not PADRAO_POL.search(produto):
                    continue
                produtos[produto.split("[")[0].strip()] += 1
            cont["retro_registros_usados"] += 1
            seq = rec.sequence.upper()
            acc = rec.name.split(":")[0]
            if len(seq) < w:
                cont["retro_registro_menor_que_janela"] += 1
                continue
            for off in starts_janela(0, len(seq), w, stride):
                sub = seq[off:off + w]
                if not so_acgt(sub):
                    cont["retro_iupac"] += 1
                    continue
                out.append(nova_janela(f"retrovirus:{rec.name}:{off}", "retrovirus", 1, sub, detail=acc,
                                       uniao=["acc:" + acc], accession=acc))
    return out, dict(sorted(produtos.items()))


# --------------------------------------------------------------------------- GC e Markov

def histograma_gc(gcs, w):
    h = [0] * N_BINS_GC
    for g in gcs:
        h[bin_gc(g, w)] += 1
    total = sum(h)
    return [x / total for x in h] if total else None


def amostrar_por_gc(cands, gc_de, n_alvo, hist, rng, w):
    """Amostra ~n_alvo de cands casando `hist`. -> (escolhidos, deficit em bins que nao encheram)."""
    por_bin = defaultdict(list)
    for c in cands:
        por_bin[bin_gc(gc_de(c), w)].append(c)
    escolhidos, deficit = [], 0
    for b in range(N_BINS_GC):
        cota = round(n_alvo * hist[b])
        disp = por_bin.get(b, [])
        if len(disp) <= cota:
            escolhidos.extend(disp)
            deficit += cota - len(disp)
        else:
            escolhidos.extend(rng.sample(disp, cota))
    return escolhidos, deficit


CODIGO = None


def _codigo():
    import numpy as np

    global CODIGO
    if CODIGO is None:
        CODIGO = np.full(256, 255, dtype=np.uint8)
        for i, b in enumerate(b"ACGT"):
            CODIGO[b] = i
    return CODIGO


def treinar_markov(seqs, k, w):
    """Contagens (4^k, 4) de k-mer -> proxima base e contagem dos k-mers iniciais. seqs: so ACGT, len w."""
    import numpy as np

    cod = _codigo()
    trans = np.zeros(4 ** k * 4, dtype=np.int64)
    iniciais = np.zeros(4 ** k, dtype=np.int64)
    for i in range(0, len(seqs), 4096):
        bloco = seqs[i:i + 4096]
        a = cod[np.frombuffer("".join(bloco).encode("ascii"), dtype=np.uint8)].astype(np.int64)
        a = a.reshape(len(bloco), w)
        idx = np.zeros((len(bloco), w - k + 1), dtype=np.int64)
        for j in range(k):
            idx = idx * 4 + a[:, j:w - k + 1 + j]
        trans += np.bincount((idx[:, :-1] * 4 + a[:, k:]).ravel(), minlength=4 ** k * 4)
        iniciais += np.bincount(idx[:, 0], minlength=4 ** k)
    return trans.reshape(4 ** k, 4), iniciais


def gerar_markov(modelo, k, w, n_alvo, hist, rng):
    """Sequencias sinteticas da cadeia, casadas em GC com `hist` por rejeicao. -> (seqs, deficit, geradas)."""
    import numpy as np

    trans, iniciais = modelo
    prob = (trans + 1) / (trans + 1).sum(axis=1, keepdims=True)  # pseudocontagem 1
    cum = np.cumsum(prob, axis=1)
    cum[:, -1] = 1.0
    p_ini = iniciais / iniciais.sum()
    cotas = [round(n_alvo * h) for h in hist]
    aceitos = [[] for _ in range(N_BINS_GC)]
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    lote = max(1000, min(2 * n_alvo, LOTE_MARKOV_MAX))
    geradas = 0
    for _ in range(MAX_LOTES_MARKOV):
        if all(len(a) >= c for a, c in zip(aceitos, cotas)):
            break
        estado = rng.choice(4 ** k, size=lote, p=p_ini)
        out = np.empty((lote, w), dtype=np.uint8)
        for j in range(k):
            out[:, j] = (estado // 4 ** (k - 1 - j)) % 4
        for pos in range(k, w):
            u = rng.random(lote)
            prox = (cum[estado] <= u[:, None]).sum(axis=1)
            out[:, pos] = prox
            estado = (estado * 4 + prox) % 4 ** k
        geradas += lote
        gcs = ((out == 1) | (out == 2)).sum(axis=1)
        for i in range(lote):
            b = bin_gc(int(gcs[i]), w)
            if len(aceitos[b]) < cotas[b]:
                aceitos[b].append(bases[out[i]].tobytes().decode("ascii"))
    deficit = sum(max(0, c - len(a)) for a, c in zip(aceitos, cotas))
    return [s for a in aceitos for s in a], deficit, geradas


# --------------------------------------------------------------------------- clustering e split

def versao_mmseqs(mmseqs):
    r = subprocess.run([mmseqs, "version"], capture_output=True, text=True)
    return r.stdout.strip() or None


def rodar_mmseqs(cmd, d):
    """Roda o MMseqs2 com o log em d/mmseqs.log; RuntimeError com a cauda do log se falhar."""
    log_mmseqs = os.path.join(d, "mmseqs.log")
    with open(log_mmseqs, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode:
        with open(log_mmseqs) as lf:
            cauda = lf.read()[-3000:]
        raise RuntimeError(f"MMseqs2 falhou ({r.returncode}): {' '.join(cmd)}\n{cauda}")


def clusterizar(janelas, mmseqs, cluster_id, tmp_base):
    """MMseqs2 easy-linclust nas duas orientacoes -> (pares (i_rep, i_membro) em indices de janelas, comando).

    Cabecalhos `s<i>_f`/`s<i>_r` (sem `|` nem `:`, que o MMseqs pode reinterpretar).
    O diretorio temporario e apagado ao final, inclusive em erro.
    """
    os.makedirs(tmp_base, exist_ok=True)
    d = tempfile.mkdtemp(prefix="mmseqs_", dir=tmp_base)
    try:
        fa = os.path.join(d, "janelas.fa")
        with open(fa, "w") as f:
            for i, j in enumerate(janelas):
                f.write(f">s{i}_f\n{j['seq']}\n>s{i}_r\n{revcomp(j['seq'])}\n")
        prefixo = os.path.join(d, "clu")
        cmd = [mmseqs, "easy-linclust", fa, prefixo, os.path.join(d, "tmp"), "--dbtype", "2",
               "--min-seq-id", str(cluster_id), "-c", str(COBERTURA_MMSEQS), "--cov-mode", "0",
               "--threads", str(THREADS_LINCLUST)]
        rodar_mmseqs(cmd, d)
        pares = []
        with open(prefixo + "_cluster.tsv") as f:
            for linha in f:
                rep, membro = linha.split()
                pares.append((int(rep[1:].split("_")[0]), int(membro[1:].split("_")[0])))
        return pares, cmd
    finally:
        shutil.rmtree(d, ignore_errors=True)


def maximos_por_consulta(linhas):
    """Linhas `query target fident alnlen qcov tcov` -> {query: [max_id (qcov >= 0.8), max_id_local (alnlen >= 200)]}.

    Consulta sem hit qualificado fica com 0.
    """
    out = {}
    for linha in linhas:
        q, _, fident, alnlen, qcov, _ = linha.rstrip("\n").split("\t")
        fident = float(fident)
        m = out.setdefault(q, [0.0, 0.0])
        if float(qcov) >= QCOV_MAX_ID:
            m[0] = max(m[0], fident)
        if int(alnlen) >= ALNLEN_MAX_ID_LOCAL:
            m[1] = max(m[1], fident)
    return out


def identidade_com_train(consultas, train, mmseqs, threads, tmp_base):
    """Uma execucao de easy-search (nucleotideo, duas fitas) de consultas x train.

    -> ({wid: (max_id_train, max_id_train_local)}, comando). Diretorio temporario apagado ao fim.
    """
    os.makedirs(tmp_base, exist_ok=True)
    d = tempfile.mkdtemp(prefix="mmseqs_busca_", dir=tmp_base)
    try:
        q_fa, t_fa, res = (os.path.join(d, n) for n in ("consultas.fa", "train.fa", "res.m8"))
        with open(q_fa, "w") as f:
            for i, j in enumerate(consultas):
                f.write(f">q{i}\n{j['seq']}\n")
        with open(t_fa, "w") as f:
            for i, j in enumerate(train):
                f.write(f">t{i}\n{j['seq']}\n")
        cmd = [mmseqs, "easy-search", q_fa, t_fa, res, os.path.join(d, "tmp"), "--search-type", "3",
               "--strand", "2", "--min-seq-id", str(MIN_ID_BUSCA), "-c", str(COBERTURA_BUSCA), "--cov-mode", "0",
               "--format-output", "query,target,fident,alnlen,qcov,tcov", "--threads", str(threads)]
        rodar_mmseqs(cmd, d)
        with open(res) as f:
            maximos = maximos_por_consulta(f)
        out = {}
        for i, j in enumerate(consultas):
            m = maximos.get(f"q{i}", (0.0, 0.0))
            out[j["wid"]] = (m[0], m[1])
        return out, cmd
    finally:
        shutil.rmtree(d, ignore_errors=True)


class UniaoBusca:
    def __init__(self):
        self.pai = {}

    def achar(self, x):
        pai = self.pai
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def unir(self, a, b):
        ra, rb = self.achar(a), self.achar(b)
        if ra != rb:
            self.pai[max(ra, rb)] = min(ra, rb)


def agrupar(janelas, pares_cluster):
    """Union-find de cluster MMseqs U `uniao` (elemento L1, accession) -> {wid: group}.

    O par fwd/rc ja e a mesma janela (mesmo wid). group = g_<sha1(menor wid)[:12]>.
    """
    uf = UniaoBusca()
    for j in janelas:
        uf.achar(j["wid"])
        for chave in j["uniao"]:
            uf.unir(j["wid"], chave)
    for a, b in pares_cluster:
        uf.unir(janelas[a]["wid"], janelas[b]["wid"])
    membros = defaultdict(list)
    for j in janelas:
        membros[uf.achar(j["wid"])].append(j["wid"])
    grupo_de = {}
    for ws in membros.values():
        g = "g_" + hashlib.sha1(min(ws).encode()).hexdigest()[:12]
        for w in ws:
            grupo_de[w] = g
    return grupo_de


def atribuir_splits(janelas, grupo_de, seed, dev_mode, dev_frac, retro_split, fracoes_markov, politica):
    """-> (split_de {wid: train/dev/test}, test_strict {wids}, dev_strict {wids}, purgadas [janelas]).

    Holdout -> test sempre. dev_mode chrom: janela genomica de dev-chrom -> dev,
    demais -> train. dev_mode hash: genomica nao-holdout segue o hash do GRUPO
    (train e dev nunca dividem grupo). Retro/markov seguem o grupo: com janela
    genomica de train -> train, senao de dev -> dev (no hash, o hash do grupo); so
    retro -> --retro-split; so markov -> fracoes_markov.
    test_strict: grupo sem janela em train/dev. dev_strict: grupo sem janela em train.
    """
    por_grupo = defaultdict(list)
    for j in janelas:
        por_grupo[grupo_de[j["wid"]]].append(j)
    split_de = {}
    for g in sorted(por_grupo):
        ms = por_grupo[g]
        u = fracao_hash(seed, g)
        genomicas = [m for m in ms if m["chrom"] and not m["holdout"]]
        if genomicas:
            if dev_mode == "hash":
                base = "dev" if u < dev_frac else "train"
            else:
                base = "train" if any(not m["dev_chrom"] for m in genomicas) else "dev"
        elif any(m["source"] == "retrovirus" for m in ms):
            base = por_fracoes(u, retro_split)
        else:
            base = por_fracoes(u, fracoes_markov)
        for m in ms:
            if m["holdout"]:
                split_de[m["wid"]] = "test"
            elif m["chrom"] and dev_mode == "chrom":
                split_de[m["wid"]] = "dev" if m["dev_chrom"] else "train"
            else:
                split_de[m["wid"]] = base

    def grupos_em(splits):
        return {g for g, ms in por_grupo.items() if any(split_de.get(m["wid"]) in splits for m in ms)}

    purgadas = []
    if politica == "purge-train":
        for g in sorted(grupos_em({"test"})):
            for m in por_grupo[g]:
                if split_de[m["wid"]] != "test":
                    purgadas.append(m)
                    del split_de[m["wid"]]
    com_trdev = grupos_em({"train", "dev"})
    com_train = grupos_em({"train"})
    test_strict = {w for w, s in split_de.items() if s == "test" and grupo_de[w] not in com_trdev}
    dev_strict = {w for w, s in split_de.items() if s == "dev" and grupo_de[w] not in com_train}
    return split_de, test_strict, dev_strict, purgadas


def balancear(janelas, mix, seed, nome):
    """Positivos + negativos em numero igual, na proporcao mix. Falta numa fonte -> completa com background.

    -> (janelas escolhidas, relatorio por fonte). Contagem em janelas (o par fwd/rc sai junto).
    """
    pos = [j for j in janelas if j["label"] == 1]
    neg = {src: sorted((j for j in janelas if j["source"] == src), key=lambda j: j["wid"]) for src in FONTES_NEG}
    for src, lst in neg.items():
        random.Random(semente(seed, "balanco", nome, src)).shuffle(lst)
    n_pos = len(pos)
    alvos = {src: round(n_pos * mix[src]) for src in ("markov", "te")}
    alvos["background"] = max(0, n_pos - alvos["markov"] - alvos["te"])  # total de negativos == positivos
    escolhidos, rel, falta = [], {}, 0
    for src in ("markov", "te"):
        usados = neg[src][:alvos[src]]
        escolhidos += usados
        deficit = alvos[src] - len(usados)
        falta += deficit
        rel[src] = {"alvo": alvos[src], "disponivel": len(neg[src]), "usados": len(usados), "deficit": deficit}
    alvo_bg = alvos["background"] + falta
    usados = neg["background"][:alvo_bg]
    escolhidos += usados
    rel["background"] = {"alvo": alvos["background"], "alvo_com_complemento": alvo_bg,
                         "disponivel": len(neg["background"]), "usados": len(usados),
                         "deficit": alvo_bg - len(usados)}
    return pos + escolhidos, rel


# --------------------------------------------------------------------------- saida e relatorio

def checar(conjuntos, grupo_de, politica, dev_mode, w):
    """Checagens que abortam antes de gravar. -> (resultado, falhas).

    Em filter-test, test dividir grupo com train/dev e (no modo chrom) dev dividir
    grupo com train sao esperados: so reportados.
    """
    wids = {c: {j["wid"] for j in js} for c, js in conjuntos.items()}
    grupos = {c: {grupo_de[x] for x in wids[c]} for c in conjuntos}
    dev_w = wids["dev"] | wids["dev_strict"]
    test_w = wids["test"] | wids["test_strict"]
    trdev_g = grupos["train"] | grupos["dev"]
    res = {
        "janela_em_train_e_outro": len(wids["train"] & (dev_w | test_w)),
        "janela_em_dev_e_test": len(dev_w & test_w),
        "grupo_em_train_e_dev": len(grupos["train"] & grupos["dev"]),
        "grupo_em_dev_strict_e_train": len(grupos["dev_strict"] & grupos["train"]),
        "grupo_em_test_strict_e_train_dev": len(grupos["test_strict"] & trdev_g),
        "grupo_em_test_e_train_dev": len(grupos["test"] & trdev_g),
        "holdout_fora_do_test": sum(1 for c in ("train", "dev", "dev_strict") for j in conjuntos[c] if j["holdout"]),
        "genomica_nao_holdout_no_test": sum(1 for c in ("test", "test_strict") for j in conjuntos[c]
                                            if j["chrom"] and not j["holdout"]),
        "dev_chrom_fora_do_dev": sum(1 for c in ("train", "test", "test_strict") for j in conjuntos[c]
                                     if j["dev_chrom"]),
        "genomica_nao_dev_chrom_no_dev": 0 if dev_mode == "hash" else sum(
            1 for c in ("dev", "dev_strict") for j in conjuntos[c] if j["chrom"] and not j["dev_chrom"]),
        "sequencia_invalida": sum(1 for js in conjuntos.values() for j in js
                                  if len(j["seq"]) != w or not so_acgt(j["seq"])),
    }
    obrigatorias = ["janela_em_train_e_outro", "janela_em_dev_e_test", "grupo_em_dev_strict_e_train",
                    "grupo_em_test_strict_e_train_dev", "holdout_fora_do_test", "genomica_nao_holdout_no_test",
                    "dev_chrom_fora_do_dev", "genomica_nao_dev_chrom_no_dev", "sequencia_invalida"]
    if dev_mode == "hash":
        obrigatorias.append("grupo_em_train_e_dev")
    if politica == "purge-train":
        obrigatorias.append("grupo_em_test_e_train_dev")
    falhas = [f"{k} = {res[k]}" for k in obrigatorias if res[k]]
    return res, falhas


def fmt_id(max_ids, wid, i):
    return "" if max_ids is None else f"{max_ids[wid][i]:.4f}"


def escrever_csv(path, janelas, grupo_de, w, max_ids=None):
    """max_ids {wid: (max_id_train, max_id_train_local)}; None (train) -> colunas vazias."""
    with open(path, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(COLUNAS)
        for j in sorted(janelas, key=lambda j: j["wid"]):
            if len(j["seq"]) != w or not so_acgt(j["seq"]):
                raise AssertionError(f"{j['wid']}: sequencia invalida (len {len(j['seq'])})")
            for ori, s in (("fwd", j["seq"]), ("rc", revcomp(j["seq"]))):
                wr.writerow([s, j["label"], j["source"], j["detail"], j["wid"], ori, j["chrom"], j["start"],
                             j["end"], j["strand"], grupo_de[j["wid"]], f"{j['gc'] / w:.4f}",
                             fmt_id(max_ids, j["wid"], 0), fmt_id(max_ids, j["wid"], 1)])


def checar_pares_csv(paths):
    """Rele os CSVs: cada window_id tem fwd+rc em cada arquivo e so divide arquivo em {dev, dev_strict} ou {test, test_strict}."""
    onde = defaultdict(lambda: defaultdict(list))
    for nome, path in paths.items():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                onde[row["window_id"]][nome].append(row["orientation"])
    separados = 0
    for por_arquivo in onde.values():
        if any(sorted(o) != ["fwd", "rc"] for o in por_arquivo.values()):
            separados += 1
        elif not any(set(por_arquivo) <= p for p in JUNTOS_PERMITIDOS):
            separados += 1
    return separados


def contagens(js):
    c = Counter((j["source"], j["label"]) for j in js)
    return {f"{src}|{lab}": c[(src, lab)] for src, lab in sorted(c)}


def gc_medio(js, w):
    if not js:
        return None
    return round(100 * sum(j["gc"] for j in js) / (len(js) * w), 2)


def resumo_gc(js, w):
    por_fonte = {src: gc_medio([j for j in js if j["source"] == src], w) for src in FONTES}
    pos = gc_medio([j for j in js if j["label"] == 1], w)
    neg = gc_medio([j for j in js if j["label"] == 0], w)
    return {"por_fonte": por_fonte, "positivos": pos, "negativos": neg,
            "diferenca_pontos": None if pos is None or neg is None else round(neg - pos, 2)}


def faixa_id(x):
    for nome, limite in FAIXAS_ID:
        if limite is None or x < limite:
            return nome


def faixas_por_fonte(js, max_ids, i):
    """{source: {faixa: n}} da metrica i (0 = max_id_train, 1 = max_id_train_local)."""
    out = {}
    for j in js:
        d = out.setdefault(j["source"], {nome: 0 for nome, _ in FAIXAS_ID})
        d[faixa_id(max_ids[j["wid"]][i])] += 1
    return {src: out[src] for src in FONTES if src in out}


def l1_fora_do_strict(todos, strict):
    """-> (por subfamilia {sf: {n, fora_do_strict, fracao}}, total {n, fora})."""
    no_strict = {j["wid"] for j in strict if j["source"] == "l1"}
    l1 = [j for j in todos if j["source"] == "l1"]
    por_sub = defaultdict(lambda: [0, 0])
    for j in l1:
        por_sub[j["subfamily"]][0] += 1
        if j["wid"] not in no_strict:
            por_sub[j["subfamily"]][1] += 1
    return ({sf: {"n": n, "fora_do_strict": f, "fracao": round(f / n, 4)} for sf, (n, f) in sorted(por_sub.items())},
            {"n": len(l1), "fora": len(l1) - len(no_strict)})


def resumo_max_id(conjuntos, max_ids):
    """Bloco max_id_train do data_meta.json (sem o comando)."""
    out = {"faixas": {}, "l1_vs_strict": {}}
    for nome in CONJUNTOS[1:]:
        out["faixas"][nome] = {"max_id_train": faixas_por_fonte(conjuntos[nome], max_ids, 0),
                               "max_id_train_local": faixas_por_fonte(conjuntos[nome], max_ids, 1)}
    for base in ("dev", "test"):
        l1 = [max_ids[j["wid"]] for j in conjuntos[base] if j["source"] == "l1"]
        out["l1_vs_strict"][base] = {
            "l1": len(l1),
            "l1_max_id_train_lt_0.90": sum(1 for g, _ in l1 if g < 0.90),
            f"l1_{base}_strict": sum(1 for j in conjuntos[f"{base}_strict"] if j["source"] == "l1"),
            "l1_max_id_train_lt_0.90_e_local_ge_0.95": sum(1 for g, loc in l1 if g < 0.90 and loc >= 0.95),
        }
    return out


def git_commit(repo):
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        sujo = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True,
                                   text=True, check=True).stdout.strip())
        return {"commit": sha, "arvore_modificada": sujo}
    except (OSError, subprocess.CalledProcessError):
        return None


def imprimir_resumo(conjuntos, w, max_id=None):
    print("\n=== Resumo (janelas; cada uma vira 2 linhas, fwd e rc) ===", flush=True)
    print(f"{'conjunto':12s} {'source':11s} {'label':>5s} {'janelas':>9s}")
    for nome in CONJUNTOS:
        c = Counter((j["source"], j["label"]) for j in conjuntos[nome])
        for src in FONTES:
            for lab in (1, 0):
                if c[(src, lab)]:
                    print(f"{nome:12s} {src:11s} {lab:5d} {c[(src, lab)]:9d}")
        print(f"{nome:12s} {'TOTAL':11s} {'':5s} {len(conjuntos[nome]):9d}")
    print(f"\n{'conjunto':12s} {'GC pos %':>9s} {'GC neg %':>9s} {'neg-pos':>8s}")
    for nome in CONJUNTOS:
        g = resumo_gc(conjuntos[nome], w)
        fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"{nome:12s} {fmt(g['positivos']):>9s} {fmt(g['negativos']):>9s} {fmt(g['diferenca_pontos']):>8s}")
    if max_id is not None:
        nomes = [n for n, _ in FAIXAS_ID]
        print(f"\n{'max_id':19s} {'conjunto':12s} {'source':11s} " + " ".join(f"{n:>9s}" for n in nomes))
        for nome in CONJUNTOS[1:]:
            for metrica in ("max_id_train", "max_id_train_local"):
                for src in ("l1", "retrovirus"):
                    f = max_id["faixas"][nome][metrica].get(src)
                    if f:
                        print(f"{metrica:19s} {nome:12s} {src:11s} " + " ".join(f"{f[n]:9d}" for n in nomes))
        for base, d in max_id["l1_vs_strict"].items():
            print(f"{base}: l1 {d['l1']}, max_id_train < 0.90: {d['l1_max_id_train_lt_0.90']} "
                  f"(l1 no {base}_strict: {d[f'l1_{base}_strict']}); max_id_train < 0.90 e local >= 0.95 "
                  f"(vazamento escondido pela cobertura): {d['l1_max_id_train_lt_0.90_e_local_ge_0.95']}")
    print(flush=True)


# --------------------------------------------------------------------------- CLI

def lista(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def parse_mix(s):
    mix = {}
    for parte in lista(s):
        k, _, v = parte.partition("=")
        mix[k.strip()] = float(v)
    if set(mix) != set(FONTES_NEG) or abs(sum(mix.values()) - 1) > 1e-6 or min(mix.values()) < 0:
        raise argparse.ArgumentTypeError(f"--neg-mix {s!r}: precisa de markov, te e background somando 1")
    return mix


def parse_fracoes(s):
    f = [float(x) for x in lista(s)]
    if len(f) != 3 or abs(sum(f) - 1) > 1e-6 or min(f) < 0:
        raise argparse.ArgumentTypeError(f"--retro-split {s!r}: 3 fracoes train,dev,test somando 1")
    return f


def construir_parser():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-identity", type=float, default=0.95)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrovirus-mode", choices=("genome", "pol"), default="genome")
    ap.add_argument("--neg-mix", type=parse_mix, default="markov=0.35,te=0.35,background=0.30")
    ap.add_argument("--te-classes", default="SINE,LINE,DNA",
                    help="repClass do rmsk para negativos te (LINE exclui L1; LTR fora por padrao)")
    ap.add_argument("--markov-order", type=int, default=5)
    ap.add_argument("--cluster-id", type=float, default=0.90)
    ap.add_argument("--holdout-chroms", default="chr8,chr21")
    ap.add_argument("--dev-mode", choices=MODOS_DEV, default="chrom",
                    help="chrom: dev = --dev-chroms; hash: dev = --dev-frac dos grupos fora do holdout")
    ap.add_argument("--dev-chroms", default="chr7,chr10")
    ap.add_argument("--dev-frac", type=float, default=0.10, help="so com --dev-mode hash")
    ap.add_argument("--retro-split", type=parse_fracoes, default="0.8,0.1,0.1")
    ap.add_argument("--leak-policy", choices=POLITICAS, default="filter-test")
    ap.add_argument("--limit-chroms", default="", help="dry run: restringe genoma e anotacoes a estes cromossomos")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                    help="threads do easy-search; o easy-linclust roda sempre com 1 (determinismo)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-if-changed", action="store_true",
                    help="pula se parametros, entradas e CSVs baterem com o data_meta.json existente")
    ap.add_argument("--hg38", default=env("HG38_FA"))
    ap.add_argument("--rmsk", default=env("RMSK_TXT"))
    ap.add_argument("--l1farm", default=env("L1FARM_TSV"))
    ap.add_argument("--retro-genomes", default=env("RETRO_GENOMES_FA"))
    ap.add_argument("--retro-cds", default=env("RETRO_CDS_FA"))
    ap.add_argument("--mmseqs", default=env("MMSEQS_BIN"))
    ap.add_argument("--tmp-dir", default=env("PREP_TMP_DIR"))
    return ap


ENTRADAS = ("hg38", "rmsk", "l1farm", "retro_genomes", "retro_cds")


def parametros(args):
    p = {k: v for k, v in vars(args).items() if k not in ("only_if_changed", "tmp_dir")}
    p["te_classes"] = lista(args.te_classes)
    p["holdout_chroms"] = lista(args.holdout_chroms)
    p["dev_chroms"] = lista(args.dev_chroms)
    p["limit_chroms"] = lista(args.limit_chroms)
    p["formato"] = FORMATO
    p["sha256_script"] = sha256_arquivo(os.path.abspath(__file__))
    return p


def dados_atuais(out_dir, params, hashes_entrada):
    try:
        with open(os.path.join(out_dir, META_NOME)) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    if meta.get("parametros") != params or meta.get("entradas_sha256") != hashes_entrada:
        return False
    for nome in CONJUNTOS:
        path = os.path.join(out_dir, f"{nome}.csv")
        if not os.path.exists(path) or sha256_arquivo(path) != meta.get("sha256", {}).get(nome):
            return False
    return True


def main(argv=None):
    import numpy as np
    import pysam

    t0 = time.time()
    tempos = {}
    marca = [time.time()]

    def etapa(nome):
        agora = time.time()
        tempos[nome] = round(agora - marca[0], 1)
        marca[0] = agora
        log(f"etapa '{nome}' concluida em {tempos[nome]} s")

    args = construir_parser().parse_args(argv)
    faltando = [n for n in ENTRADAS + ("mmseqs", "tmp_dir") if not getattr(args, n)]
    if faltando:
        sys.exit(f"ERRO: faltam {', '.join('--' + n.replace('_', '-') for n in faltando)}; passe por CLI ou "
                 "carregue environments/config.local.sh")
    for n in ENTRADAS:
        if not os.path.exists(getattr(args, n)):
            sys.exit(f"ERRO: --{n.replace('_', '-')} {getattr(args, n)} nao existe")
    w = args.window
    if w % 128:
        sys.exit(f"ERRO: --window {w} nao e multiplo de 128 (exigencia do NT v3)")
    if not 0 < args.dev_frac < 1:
        sys.exit(f"ERRO: --dev-frac {args.dev_frac} fora de (0, 1)")
    limite = lista(args.limit_chroms)
    dev_chroms = lista(args.dev_chroms) if args.dev_mode == "chrom" else []
    invalidos = [c for c in limite + lista(args.holdout_chroms) + dev_chroms if c not in PRIMARIOS]
    if invalidos:
        sys.exit(f"ERRO: cromossomos fora de chr1-22/X/Y: {invalidos}")
    comuns = sorted(set(dev_chroms) & set(lista(args.holdout_chroms)))
    if comuns:
        sys.exit(f"ERRO: --dev-chroms e --holdout-chroms precisam ser disjuntos; em comum: {comuns}")

    params = parametros(args)
    log(f"hasheando entradas ({', '.join(ENTRADAS)})")
    hashes_entrada = {n: sha256_arquivo(getattr(args, n)) for n in ENTRADAS}
    if args.only_if_changed and dados_atuais(args.out_dir, params, hashes_entrada):
        log(f"dados atuais em {args.out_dir} (parametros, entradas e CSVs batem), nada a fazer")
        return
    etapa("hash_entradas")

    cont = Counter()
    fa = pysam.FastaFile(args.hg38)
    presentes = set(fa.references)
    cromossomos = [c for c in PRIMARIOS if c in presentes and (not limite or c in limite)]
    holdout = set(lista(args.holdout_chroms)) & set(cromossomos)
    dev_set = set(dev_chroms) & set(cromossomos)
    if not holdout:
        log("AVISO: nenhum cromossomo de holdout entre os processados; o test so tera retrovirus/markov")
    if args.dev_mode == "chrom" and not dev_set:
        log("AVISO: nenhum cromossomo de dev entre os processados; o dev so tera retrovirus/markov")
    log(f"cromossomos: {','.join(cromossomos)} | holdout: {','.join(sorted(holdout)) or '-'} | "
        f"dev ({args.dev_mode}): {','.join(sorted(dev_set)) or '-'}")

    def papel(chrom):
        return "test" if chrom in holdout else ("dev" if chrom in dev_set else "train")

    regioes, fora = ler_l1farm(args.l1farm, set(cromossomos))
    filtradas = [r for r in regioes if r["identity"] >= args.min_identity]
    elementos = agrupar_elementos(filtradas)
    log(f"L1Farm: {len(regioes)} regioes nos cromossomos ({fora} fora), {len(filtradas)} com Identity >= "
        f"{args.min_identity}, {len(elementos)} elementos")
    te_classes = set(lista(args.te_classes))
    rmsk = ler_rmsk(args.rmsk, set(cromossomos), te_classes)
    log("rmsk: " + ", ".join(f"{k} {sum(len(d[k]) for d in rmsk.values())}"
                             for k in ("l1", "ltr_sva", "intercalado", "te")))
    etapa("anotacoes")

    # Passada A: l1 com sequencia; te e background so coordenada + GC
    l1, cand_te, cand_bg = [], [], []
    regioes_por_chrom, elementos_por_chrom, todas_por_chrom = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in filtradas:
        regioes_por_chrom[r["chrom"]].append(r)
    for r in regioes:
        todas_por_chrom[r["chrom"]].append((r["start"], r["end"]))
    for el in elementos:
        elementos_por_chrom[el["chrom"]].append(el)
    for chrom in cromossomos:
        seq = fa.fetch(chrom).upper()
        d = rmsk[chrom]
        l1_any = Intervalos(d["l1"] + todas_por_chrom[chrom])
        excl = Intervalos(d["l1"] + todas_por_chrom[chrom] + d["ltr_sva"])
        bloqueio = Intervalos(d["intercalado"] + d["l1"] + d["ltr_sva"] + todas_por_chrom[chrom])
        pp = papel(chrom)
        js = extrair_l1(chrom, seq, elementos_por_chrom[chrom], regioes_por_chrom[chrom], w, args.stride,
                        pp == "test", cont)
        for j in js:
            j["dev_chrom"] = pp == "dev"
        l1 += js
        te = candidatos_te(seq, d["te"], l1_any, excl, w, cont)
        cand_te += [(chrom, pp) + t for t in te]
        bg = candidatos_background(seq, bloqueio, w, cont)
        cand_bg += [(chrom, pp) + b for b in bg]
        log(f"{chrom}: l1 {len(js)}, candidatos te {len(te)}, background {len(bg)}")
        del seq
    etapa("passada_A")

    retro, produtos_pol = janelas_retrovirus(args.retro_genomes, args.retro_cds, args.retrovirus_mode, w,
                                             args.stride, cont)
    log(f"retrovirus ({args.retrovirus_mode}): {cont['retro_registros_usados']} registros, {len(retro)} janelas")
    if args.retrovirus_mode == "pol":
        log(f"CDS pol que entraram: {cont['retro_registros_usados']} ({produtos_pol})")
    etapa("retrovirus")

    l1_por_papel = {pp: [j for j in l1 if papel(j["chrom"]) == pp] for pp in ("train", "dev", "test")}
    l1_train = l1_por_papel["train"]
    if not l1_train:
        sys.exit("ERRO: nenhuma janela l1 nos cromossomos de train (necessaria para treinar o Markov e o train)")
    hist_global = histograma_gc([j["gc"] for j in l1], w)
    hists = {pp: histograma_gc([j["gc"] for j in js], w) or hist_global for pp, js in l1_por_papel.items()}
    r_train, r_dev, r_test = (len(retro) * f for f in args.retro_split)
    p_est = {"train": len(l1_train) + r_train, "dev": len(l1_por_papel["dev"]) + r_dev,
             "test": len(l1_por_papel["test"]) + r_test}
    p_total = sum(p_est.values())
    if args.dev_mode == "chrom":
        fracoes_markov = [p_est[pp] / p_total for pp in ("train", "dev", "test")]
        pools = ("test", "dev", "train")
    else:  # o dev sai do pool de train pelo hash do grupo
        p_est["train"] += p_est.pop("dev")
        p_trdev = p_est["train"]
        fracoes_markov = [(p_trdev / p_total) * (1 - args.dev_frac), (p_trdev / p_total) * args.dev_frac,
                          p_est["test"] / p_total]
        pools = ("test", "train")
    mix = args.neg_mix

    deficits_pool = {}
    escolhidos_te, escolhidos_bg = [], []
    for pool in pools:
        fator_te = OVERSAMPLE if pool == "train" else OVERSAMPLE_TE_STRICT
        n_te = int(np.ceil(p_est[pool] * mix["te"] * fator_te))
        n_bg = int(np.ceil(p_est[pool] * OVERSAMPLE))  # background cobre tambem os deficits das outras fontes
        hist = hists[pool]
        rng = random.Random(semente(args.seed, "pool", pool))
        te_pool, def_te = amostrar_por_gc([c for c in cand_te if c[1] == pool], lambda c: c[-1], n_te, hist, rng, w)
        bg_pool, def_bg = amostrar_por_gc([c for c in cand_bg if c[1] == pool], lambda c: c[-1], n_bg, hist, rng, w)
        escolhidos_te += te_pool
        escolhidos_bg += bg_pool
        deficits_pool[pool] = {"te": {"alvo": n_te, "obtido": len(te_pool), "deficit_bins_gc": def_te},
                               "background": {"alvo": n_bg, "obtido": len(bg_pool), "deficit_bins_gc": def_bg}}
        log(f"pool {pool}: te {len(te_pool)}/{n_te}, background {len(bg_pool)}/{n_bg} (deficit em bins de GC: "
            f"te {def_te}, background {def_bg})")
    n_cand = {"te": len(cand_te), "background": len(cand_bg)}
    del cand_te, cand_bg

    # Passada B: sequencia so dos negativos sorteados
    negativos = []
    por_chrom = defaultdict(list)
    for c in escolhidos_te:
        por_chrom[c[0]].append(("te", c))
    for c in escolhidos_bg:
        por_chrom[c[0]].append(("background", c))
    for chrom in cromossomos:
        if not por_chrom[chrom]:
            continue
        seq = fa.fetch(chrom).upper()
        for src, c in sorted(por_chrom[chrom], key=lambda x: (x[0], x[1][2])):
            s = c[2]
            sub = seq[s:s + w]
            pp = c[1]
            if src == "te":
                _, _, _, strand, nome, classe, familia, _ = c
                negativos.append(nova_janela(f"te:{chrom}:{s}-{s + w}", "te", 0, sub, chrom=chrom, start=s,
                                             end=s + w, strand=strand, detail=f"{nome}/{classe}",
                                             holdout=pp == "test", dev_chrom=pp == "dev", rep_family=familia))
            else:
                negativos.append(nova_janela(f"background:{chrom}:{s}-{s + w}", "background", 0, sub,
                                             chrom=chrom, start=s, end=s + w, strand=".", holdout=pp == "test",
                                             dev_chrom=pp == "dev"))
        del seq
    fa.close()
    etapa("negativos_genomicos")

    k = args.markov_order
    modelo = treinar_markov([j["seq"] for j in l1_train] + [revcomp(j["seq"]) for j in l1_train], k, w)
    n_markov = int(np.ceil(p_total * mix["markov"] * OVERSAMPLE))
    seqs_mk, def_mk, geradas = gerar_markov(modelo, k, w, n_markov, hists["train"],
                                            np.random.default_rng(semente(args.seed, "markov")))
    markov = [nova_janela(f"markov:{i:07d}", "markov", 0, s, detail=f"order{k}") for i, s in enumerate(seqs_mk)]
    deficits_pool["markov"] = {"alvo": n_markov, "obtido": len(markov), "deficit_bins_gc": def_mk,
                               "geradas": geradas}
    log(f"markov ordem {k}: {len(markov)}/{n_markov} aceitas de {geradas} geradas (deficit em bins de GC {def_mk})")
    etapa("markov")

    janelas = sorted(l1 + retro + negativos + markov, key=lambda j: j["wid"])
    if len({j["wid"] for j in janelas}) != len(janelas):
        sys.exit("ERRO: window_id duplicado")
    log(f"clustering de {len(janelas)} janelas ({2 * len(janelas)} sequencias) com MMseqs2")
    pares, cmd = clusterizar(janelas, args.mmseqs, args.cluster_id, args.tmp_dir)
    grupo_de = agrupar(janelas, pares)
    tamanho = Counter(grupo_de.values())
    log(f"{len(tamanho)} grupos; maior com {max(tamanho.values())} janelas")
    etapa("clustering")

    split_de, test_strict, dev_strict, purgadas = atribuir_splits(
        janelas, grupo_de, args.seed, args.dev_mode, args.dev_frac, args.retro_split, fracoes_markov,
        args.leak_policy)
    purge_fonte = Counter(j["source"] for j in purgadas)
    purge_l1_sub = Counter(j["subfamily"] for j in purgadas if j["source"] == "l1")
    if purgadas:
        n_l1_fora = sum(1 for j in l1 if not j["holdout"])
        log(f"!!! PURGE: {len(purgadas)} janelas removidas por fonte {dict(purge_fonte)}; l1 de treino removido: "
            f"{sum(purge_l1_sub.values())}/{n_l1_fora} ({dict(purge_l1_sub.most_common())})")
    pre = {nome: [] for nome in CONJUNTOS}
    for j in janelas:
        s = split_de.get(j["wid"])
        if s:
            pre[s].append(j)
            if j["wid"] in test_strict:
                pre["test_strict"].append(j)
            if j["wid"] in dev_strict:
                pre["dev_strict"].append(j)
    conjuntos, balanco = {}, {}
    for nome in CONJUNTOS:
        if ((nome == "test_strict" and args.leak_policy == "purge-train")
                or (nome == "dev_strict" and args.dev_mode == "hash")):
            base = nome[:-len("_strict")]  # sem grupo compartilhado: *_strict == base
            conjuntos[nome], balanco[nome] = conjuntos[base], balanco[base]
            continue
        conjuntos[nome], balanco[nome] = balancear(pre[nome], mix, args.seed, nome)
        deficits = {src: r["deficit"] for src, r in balanco[nome].items() if r["deficit"]}
        if deficits:
            log(f"AVISO: deficit de negativos em {nome}: {deficits} (te/markov completados com background)")
    etapa("split_e_balanco")

    res_checagens, falhas = checar(conjuntos, grupo_de, args.leak_policy, args.dev_mode, w)
    if falhas:
        sys.exit("ERRO: checagens falharam, nada foi gravado:\n  " + "\n  ".join(falhas))

    vistos = set()
    consultas = []
    for nome in CONJUNTOS[1:]:
        for j in conjuntos[nome]:
            if j["wid"] not in vistos:
                vistos.add(j["wid"])
                consultas.append(j)
    consultas.sort(key=lambda j: j["wid"])
    train_ord = sorted(conjuntos["train"], key=lambda j: j["wid"])
    log(f"identidade com o train: {len(consultas)} janelas de dev/test x {len(train_ord)} do train (easy-search)")
    max_ids, cmd_busca = identidade_com_train(consultas, train_ord, args.mmseqs, args.threads, args.tmp_dir)
    max_id = resumo_max_id(conjuntos, max_ids)
    etapa("identidade_train")

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_paths = {nome: os.path.join(args.out_dir, f"{nome}.csv.tmp") for nome in CONJUNTOS}
    for nome in CONJUNTOS:
        escrever_csv(tmp_paths[nome], conjuntos[nome], grupo_de, w, None if nome == "train" else max_ids)
    separados = checar_pares_csv(tmp_paths)
    res_checagens["pares_fwd_rc_separados_ou_janela_em_dois_splits"] = separados
    if separados:
        sys.exit(f"ERRO: {separados} window_id com par fwd/rc separado ou em dois splits (arquivos .tmp mantidos)")
    meta_path = os.path.join(args.out_dir, META_NOME)
    if os.path.exists(meta_path):
        os.remove(meta_path)
    hashes = {}
    for nome in CONJUNTOS:
        final = os.path.join(args.out_dir, f"{nome}.csv")
        os.replace(tmp_paths[nome], final)
        hashes[nome] = sha256_arquivo(final)
    etapa("escrita")

    # Vazamento: l1 de test/dev fora do *_strict, por subfamilia
    fora_test_sub, fora_test = l1_fora_do_strict(conjuntos["test"], conjuntos["test_strict"])
    fora_dev_sub, fora_dev = l1_fora_do_strict(conjuntos["dev"], conjuntos["dev_strict"])
    g_maior, n_maior = min(tamanho.items(), key=lambda kv: (-kv[1], kv[0]))
    membros_maior = [j for j in janelas if grupo_de[j["wid"]] == g_maior]
    n_l1 = len(l1)

    meta = {
        "parametros": params,
        "entradas": {n: getattr(args, n) for n in ENTRADAS},
        "entradas_sha256": hashes_entrada,
        "hg38_fa_gz_md5_ucsc": "1c9dcaddfa41027f17cd8f7a82c7293b",
        "git": git_commit(os.path.dirname(os.path.abspath(__file__))),
        "versoes": {"python": platform.python_version(), "pysam": pysam.__version__, "numpy": np.__version__,
                    "mmseqs2": versao_mmseqs(args.mmseqs)},
        "mmseqs_comando": " ".join(cmd),
        "cromossomos": cromossomos,
        "holdout_presentes": sorted(holdout),
        "dev_chroms_presentes": sorted(dev_set),
        "cromossomos_por_conjunto": {n: sorted({j["chrom"] for j in js if j["chrom"]}, key=PRIMARIOS.index)
                                     for n, js in conjuntos.items()},
        "contagens_janelas": {n: contagens(js) for n, js in conjuntos.items()},
        "contagens_linhas": {n: 2 * len(js) for n, js in conjuntos.items()},
        "gc_medio_pct": {n: resumo_gc(js, w) for n, js in conjuntos.items()},
        "te_por_repfamily": {n: dict(sorted(Counter(j["rep_family"] for j in js if j["source"] == "te").items()))
                             for n, js in conjuntos.items()},
        "l1_por_subfamilia": {n: dict(sorted(Counter(j["subfamily"] for j in js if j["source"] == "l1").items()))
                              for n, js in conjuntos.items()},
        "vazamento": {
            "politica": args.leak_policy,
            "dev_mode": args.dev_mode,
            "nota": "test_strict e a metrica principal; test e secundaria" if args.leak_policy == "filter-test"
                    else "purge-train: test_strict == test",
            "n_grupos": len(tamanho),
            "maior_grupo": {"group": g_maior, "janelas": n_maior,
                            "por_fonte": dict(sorted(Counter(j["source"] for j in membros_maior).items())),
                            "fracao_do_l1": round(sum(1 for j in membros_maior if j["source"] == "l1") / n_l1, 4)
                            if n_l1 else None},
            "l1_test_fora_do_test_strict_por_subfamilia": fora_test_sub,
            "l1_test_fora_do_test_strict_total": fora_test,
            "l1_dev_fora_do_dev_strict_por_subfamilia": fora_dev_sub,
            "l1_dev_fora_do_dev_strict_total": fora_dev,
            "purge_por_fonte": dict(sorted(purge_fonte.items())),
            "purge_l1_por_subfamilia": dict(sorted(purge_l1_sub.items())),
        },
        "max_id_train": {"comando": " ".join(cmd_busca),
                         "definicao": {"max_id_train": f"maior fident com qcov >= {QCOV_MAX_ID}",
                                       "max_id_train_local": f"maior fident com alnlen >= {ALNLEN_MAX_ID_LOCAL}",
                                       "sem_hit": 0},
                         **max_id},
        "descartes": dict(sorted(cont.items())),
        "candidatos_negativos": n_cand,
        "deficits_amostragem": deficits_pool,
        "balanco": balanco,
        "fracoes_markov_estimadas": [round(x, 4) for x in fracoes_markov],
        "produtos_pol": produtos_pol if args.retrovirus_mode == "pol" else None,
        "checagens": res_checagens,
        "sha256": hashes,
        "tempo_s": {**tempos, "total": round(time.time() - t0, 1)},
        "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    imprimir_resumo(conjuntos, w, max_id)
    for base, tot in (("test", fora_test), ("dev", fora_dev)):
        if tot["n"]:
            log(f"!!! VAZAMENTO: {tot['fora']}/{tot['n']} janelas l1 de {base} fora do {base}_strict "
                f"({100 * tot['fora'] / tot['n']:.1f}%)")
    log(f"maior grupo = {n_maior} janelas ({meta['vazamento']['maior_grupo']['fracao_do_l1']} do l1). "
        "Por subfamilia em data_meta.json")
    log(f"saida: {args.out_dir} | tempo total {meta['tempo_s']['total']} s")


if __name__ == "__main__":
    main()
