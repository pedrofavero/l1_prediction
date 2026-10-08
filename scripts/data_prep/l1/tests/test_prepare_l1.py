"""Testes do prepare_l1_dataset.py com FASTA/TSV sinteticos gerados aqui mesmo.

Rodar: "$PREP_PYTHON" -m unittest discover scripts/data_prep/l1/tests
O teste ponta a ponta precisa do MMseqs2 (MMSEQS_BIN ou o caminho padrao do
config.local.sh); sem ele, e pulado.
"""
import contextlib
import csv
import gzip
import importlib.util
import io
import os
import random
import tempfile
import unittest
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("prepare_l1_dataset",
                                               os.path.join(AQUI, "..", "prepare_l1_dataset.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

MMSEQS = os.environ.get("MMSEQS_BIN") or os.path.expanduser("~/l1_data/bin/mmseqs/bin/mmseqs")
W = 1024


def seq_aleatoria(n, rng, gc=0.41):
    return "".join(rng.choices("ACGT", weights=[(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2], k=n))


def mutar(s, taxa, rng):
    return "".join(rng.choice("ACGT".replace(b, "")) if rng.random() < taxa else b for b in s)


def escrever_fasta(path, registros, largura=60):
    with open(path, "w") as f:
        for nome, seq in registros:
            f.write(f">{nome}\n")
            for i in range(0, len(seq), largura):
                f.write(seq[i:i + largura] + "\n")


def linha_l1farm(chrom, s, e, sub, reg, strand, ident):
    return "\t".join(map(str, (chrom, s, e, sub, reg, strand, 0, 0, 0, e - s, ident, 1))) + "\n"


def linha_rmsk(chrom, s, e, strand, nome, classe, familia):
    return "\t".join(map(str, (0, 0, 0, 0, 0, chrom, s, e, 0, strand, nome, classe, familia, 0, 0, 0, 1))) + "\n"


def janela(wid, source, label, chrom="", holdout=False, uniao=(), dev_chrom=False):
    return P.nova_janela(wid, source, label, "A" * W, chrom=chrom, holdout=holdout, uniao=list(uniao),
                         dev_chrom=dev_chrom)


FR = [0.8, 0.1, 0.1]


class TestBasicos(unittest.TestCase):
    def test_revcomp(self):
        self.assertEqual(P.revcomp("AACGTG"), "CACGTT")
        self.assertEqual(P.revcomp(P.revcomp("ACGTTGCA")), "ACGTTGCA")

    def test_intervalos(self):
        iv = P.Intervalos([(15, 30), (10, 20), (40, 50)])
        self.assertEqual(iv.sobreposicao(0, 100), 30)
        self.assertEqual(iv.sobreposicao(25, 45), 10)
        self.assertEqual(iv.sobreposicao(30, 40), 0)   # half-open: 30 e 40 nao estao cobertos
        self.assertEqual(iv.sobreposicao(29, 41), 2)
        self.assertEqual(list(iv.lacunas(0, 60)), [(0, 10), (30, 40), (50, 60)])

    def test_bin_gc(self):
        self.assertEqual(P.bin_gc(0, W), 0)
        self.assertEqual(P.bin_gc(W, W), P.N_BINS_GC - 1)
        self.assertEqual(P.bin_gc(512, 1024), 25)

    def test_faixa_id(self):
        self.assertEqual([P.faixa_id(x) for x in (0, 0.7999, 0.80, 0.8999, 0.90, 0.9499, 0.95, 1.0)],
                         ["<0.80", "<0.80", "0.80-0.90", "0.80-0.90", "0.90-0.95", "0.90-0.95", ">=0.95", ">=0.95"])

    def test_maximos_por_consulta(self):
        linhas = ["q0\tt1\t0.990\t300\t0.29\t0.29\n",    # local, sem cobertura
                  "q0\tt2\t0.930\t1000\t0.98\t0.98\n",   # cobertura
                  "q0\tt3\t0.910\t1024\t1.00\t1.00\n",
                  "q1\tt1\t0.999\t150\t0.19\t0.19\n"]    # curto demais e sem cobertura
        m = P.maximos_por_consulta(linhas)
        self.assertEqual(m["q0"], [0.93, 0.99])
        self.assertEqual(m["q1"], [0.0, 0.0])
        self.assertNotIn("q2", m)  # sem hit -> identidade_com_train usa 0

    def test_starts_janela(self):
        self.assertEqual(P.starts_janela(1000, 1300, W, 512), [1150 - 512])         # centrada
        self.assertEqual(P.starts_janela(100, 300, W, 512), [0])                     # encosta no 0
        self.assertEqual(P.starts_janela(0, 2500, W, 512), [0, 512, 1024, 1476])     # + janela final
        self.assertEqual(P.starts_janela(0, 2048, W, 512), [0, 512, 1024])
        self.assertEqual(P.starts_janela(4900, 5000, W, 512, limite=5000), [5000 - W])


class TestLeitura(unittest.TestCase):
    def test_l1farm_comentarios_brancos_e_coordenadas(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l1farm.tsv")
            with open(path, "w") as f:
                f.write("## L1Farm\n# subset\n\n   \n# ref\n\n")
                f.write(linha_l1farm("chr1", 100, 200, "L1HS", "5UTR", "+", 0.99))
                f.write(linha_l1farm("chr1_KI270706v1_random", 5, 50, "L1HS", "3UTR", "-", 0.99))
                f.write("\n")
            regioes, fora = P.ler_l1farm(path, {"chr1"})
            self.assertEqual(fora, 1)
            self.assertEqual(len(regioes), 1)
            r = regioes[0]
            self.assertEqual((r["start"], r["end"]), (100, 200))  # 0-based, sem Start-1
            self.assertAlmostEqual(r["identity"], 0.99)

            with open(path, "w") as f:
                f.write("chr1\t100\t200\tL1HS\t5UTR\t+\t0\t0\t0\t101\t0.99\t1\n")  # LocusLength de 1-based
            with self.assertRaises(ValueError):
                P.ler_l1farm(path, {"chr1"})

    def test_agrupar_elementos(self):
        rs = [{"chrom": "chr1", "start": 0, "end": 100, "strand": "+"},
              {"chrom": "chr1", "start": 600, "end": 700, "strand": "+"},    # gap 500 -> mesmo elemento
              {"chrom": "chr1", "start": 1201, "end": 1300, "strand": "+"},  # gap 501 -> novo
              {"chrom": "chr1", "start": 650, "end": 690, "strand": "-"}]    # outra fita -> novo
        els = P.agrupar_elementos(rs)
        spans = sorted((e["strand"], e["start"], e["end"]) for e in els)
        self.assertEqual(spans, [("+", 0, 700), ("+", 1201, 1300), ("-", 650, 690)])
        self.assertIn("chr1:+:0-700", {e["element_id"] for e in els})


class TestJanelasGenomicas(unittest.TestCase):
    def test_extrair_l1_upper_coordenadas_e_rotulo(self):
        import pysam

        rng = random.Random(1)
        orig = seq_aleatoria(20000, rng)
        orig = orig[:12100] + "N" + orig[12101:]
        soft = orig[:1500] + orig[1500:3000].lower() + orig[3000:]  # soft-mask
        with tempfile.TemporaryDirectory() as d:
            fa_path = os.path.join(d, "g.fa")
            escrever_fasta(fa_path, [("chrT", soft)])
            pysam.faidx(fa_path)
            with pysam.FastaFile(fa_path) as fa:
                seq = fa.fetch("chrT").upper()
                self.assertEqual(fa.fetch("chrT", 1638, 1638 + W).upper(), orig[1638:1638 + W])
            regs = [{"chrom": "chrT", "start": 2000, "end": 2300, "subfamily": "L1HS", "region": "5UTR",
                     "strand": "+", "identity": 0.99},   # 300 bp -> positivo
                    {"chrom": "chrT", "start": 8000, "end": 8150, "subfamily": "L1PA2", "region": "3UTR",
                     "strand": "-", "identity": 0.99},   # 150 bp -> zona cinza
                    {"chrom": "chrT", "start": 12000, "end": 12400, "subfamily": "L1PA3", "region": "ORF1",
                     "strand": "+", "identity": 0.99}]   # tem N -> descarte
            cont = P.Counter()
            js = P.extrair_l1("chrT", seq, P.agrupar_elementos(regs), regs, W, 512, False, cont)
        self.assertEqual(len(js), 1)
        j = js[0]
        s = 2150 - W // 2
        self.assertEqual((j["start"], j["end"], j["wid"]), (s, s + W, f"l1:chrT:{s}-{s + W}"))
        self.assertEqual(j["seq"], orig[s:s + W])  # maiusculas, fatia 0-based
        self.assertTrue(P.so_acgt(j["seq"]))
        self.assertEqual((j["detail"], j["subfamily"], j["strand"]), ("L1HS:5UTR", "L1HS", "+"))
        self.assertEqual(cont["l1_zona_cinza_1a199bp"], 1)
        self.assertEqual(cont["l1_com_N"], 1)

    def test_rotulo_por_sobreposicao_limiar_200(self):
        seq = "ACGT" * 2000
        for bp, esperado in ((199, 0), (200, 1)):
            regs = [{"chrom": "c", "start": 3000, "end": 3000 + bp, "subfamily": "L1HS", "region": "3UTR",
                     "strand": "+", "identity": 1.0}]
            js = P.extrair_l1("c", seq, P.agrupar_elementos(regs), regs, W, 512, False, P.Counter())
            self.assertEqual(len(js), esperado, bp)

    def test_negativos_zona_cinza_e_exclusoes(self):
        seq = "ACGT" * 10000
        l1_any = P.Intervalos([(5000, 5100)])                 # L1 antigo
        excl = P.Intervalos([(5000, 5100), (20000, 20100)])   # + LTR
        tes = [(5300, 5600, "+", "AluY", "SINE", "Alu"),      # janela toca o L1 antigo
               (20200, 20400, "+", "MIR", "SINE", "MIR"),     # janela toca o LTR
               (30000, 30300, "-", "L2a", "LINE", "L2")]
        cont = P.Counter()
        out = P.candidatos_te(seq, tes, l1_any, excl, W, cont)
        self.assertEqual([c[0] for c in out], [30150 - W // 2])
        self.assertEqual(cont["neg_zona_cinza_toca_l1"], 1)
        self.assertEqual(cont["te_toca_ltr_sva"], 1)
        bg = P.candidatos_background(seq, P.Intervalos([(0, 5000), (7000, 40000)]), W, P.Counter())
        self.assertEqual([b[0] for b in bg], [5000])  # so cabe uma janela em [5000, 7000)


class TestSplit(unittest.TestCase):
    def _janelas(self):
        js = [janela("l1:chr21:0-1024", "l1", 1, "chr21", True, ["elem:A"]),
              janela("l1:chr1:0-1024", "l1", 1, "chr1", False, ["elem:B"]),     # cluster com o de cima
              janela("l1:chr21:5000-6024", "l1", 1, "chr21", True, ["elem:C"]),  # sozinho -> strict
              janela("l1:chr1:9000-10024", "l1", 1, "chr1", False, ["elem:D"]),
              janela("l1:chr1:9512-10536", "l1", 1, "chr1", False, ["elem:D"]),  # mesmo elemento
              janela("retrovirus:NC_1.1:0", "retrovirus", 1, uniao=["acc:NC_1.1"]),
              janela("retrovirus:NC_1.1:512", "retrovirus", 1, uniao=["acc:NC_1.1"])]
        pares = [(0, 1)]
        return js, P.agrupar(js, pares)

    def test_union_find(self):
        js, g = self._janelas()
        self.assertEqual(g["l1:chr21:0-1024"], g["l1:chr1:0-1024"])
        self.assertEqual(g["l1:chr1:9000-10024"], g["l1:chr1:9512-10536"])
        self.assertEqual(g["retrovirus:NC_1.1:0"], g["retrovirus:NC_1.1:512"])
        self.assertNotEqual(g["l1:chr21:5000-6024"], g["l1:chr21:0-1024"])

    def test_filter_test(self):
        js, g = self._janelas()
        split, strict, _, purg = P.atribuir_splits(js, g, 42, "hash", 0.1, FR, FR, "filter-test")
        self.assertEqual(purg, [])
        self.assertEqual(len(split), len(js))                       # train/dev intactos
        self.assertEqual(split["l1:chr21:0-1024"], "test")
        self.assertIn(split["l1:chr1:0-1024"], ("train", "dev"))
        self.assertNotIn("l1:chr21:0-1024", strict)                 # grupo tem janela em train/dev
        self.assertIn("l1:chr21:5000-6024", strict)
        self.assertEqual(split["l1:chr1:9000-10024"], split["l1:chr1:9512-10536"])
        self.assertEqual(split["retrovirus:NC_1.1:0"], split["retrovirus:NC_1.1:512"])

    def test_purge_train(self):
        js, g = self._janelas()
        split, strict, _, purg = P.atribuir_splits(js, g, 42, "hash", 0.1, FR, FR, "purge-train")
        self.assertEqual([j["wid"] for j in purg], ["l1:chr1:0-1024"])
        self.assertNotIn("l1:chr1:0-1024", split)
        self.assertEqual(strict, {w for w, s in split.items() if s == "test"})

    def test_grupo_nunca_dividido_entre_train_e_dev(self):
        rng = random.Random(0)
        js = [janela(f"l1:chr1:{i}-{i + W}", "l1", 1, "chr1", False, [f"elem:{i // 7}"]) for i in range(400)]
        pares = [(rng.randrange(400), rng.randrange(400)) for _ in range(60)]
        g = P.agrupar(js, pares)
        split, _, dev_strict, _ = P.atribuir_splits(js, g, 42, "hash", 0.3, FR, FR, "filter-test")
        por_grupo = defaultdict(set)
        for wid, s in split.items():
            por_grupo[g[wid]].add(s)
        self.assertTrue(all(len(s) == 1 for s in por_grupo.values()))
        self.assertEqual(set(split.values()), {"train", "dev"})
        self.assertEqual(dev_strict, {w for w, s in split.items() if s == "dev"})  # hash: dev_strict == dev

    def test_dev_por_cromossomo(self):
        js = [janela("l1:chr21:0-1024", "l1", 1, "chr21", True, ["elem:A"]),
              janela("l1:chr10:0-1024", "l1", 1, "chr10", False, ["elem:B"], dev_chrom=True),
              janela("l1:chr1:0-1024", "l1", 1, "chr1", False, ["elem:C"]),        # mesmo grupo dos 2 acima
              janela("l1:chr10:5000-6024", "l1", 1, "chr10", False, ["elem:D"], dev_chrom=True),  # sozinho
              janela("l1:chr10:9000-10024", "l1", 1, "chr10", False, ["elem:E"], dev_chrom=True),
              janela("l1:chr21:9000-10024", "l1", 1, "chr21", True, ["elem:F"]),   # grupo so com dev
              janela("retrovirus:NC_1.1:0", "retrovirus", 1, uniao=["acc:NC_1.1"]),
              janela("markov:0000001", "markov", 0)]
        g = P.agrupar(js, [(0, 1), (1, 2), (4, 5), (7, 3)])  # markov no grupo de um dev
        split, test_strict, dev_strict, purg = P.atribuir_splits(js, g, 42, "chrom", 0.1, FR, FR, "filter-test")
        self.assertEqual(purg, [])
        self.assertEqual(len(split), len(js))  # train intacto, nada removido
        self.assertEqual([split[w] for w in ("l1:chr21:0-1024", "l1:chr10:0-1024", "l1:chr1:0-1024")],
                         ["test", "dev", "train"])
        self.assertEqual(split["markov:0000001"], "dev")            # segue a genomica do grupo
        self.assertEqual(split["retrovirus:NC_1.1:0"], P.por_fracoes(P.fracao_hash(42, g["retrovirus:NC_1.1:0"]), FR))
        self.assertEqual(dev_strict, {"l1:chr10:5000-6024", "l1:chr10:9000-10024", "markov:0000001"})
        self.assertNotIn("l1:chr21:9000-10024", test_strict)        # grupo com dev
        self.assertNotIn("l1:chr21:0-1024", test_strict)

    def test_dev_e_holdout_disjuntos(self):
        with tempfile.TemporaryDirectory() as d:
            vazios = []
            for nome in ("g.fa", "r.gz", "l.tsv", "v.fa", "c.fa"):
                open(os.path.join(d, nome), "w").close()
                vazios.append(os.path.join(d, nome))
            argv = ["--out-dir", d, "--hg38", vazios[0], "--rmsk", vazios[1], "--l1farm", vazios[2],
                    "--retro-genomes", vazios[3], "--retro-cds", vazios[4], "--mmseqs", "x", "--tmp-dir", d,
                    "--dev-chroms", "chr7,chr21"]
            with self.assertRaises(SystemExit) as ctx:
                P.main(argv)
            self.assertIn("disjuntos", str(ctx.exception.code))

    def test_balancear_completa_com_background(self):
        mix = {"markov": 0.35, "te": 0.35, "background": 0.30}
        js = ([janela(f"l1:c:{i}", "l1", 1) for i in range(20)] + [janela("te:c:0", "te", 0)] +
              [janela(f"markov:{i}", "markov", 0) for i in range(30)] +
              [janela(f"background:c:{i}", "background", 0) for i in range(40)])
        esc, rel = P.balancear(js, mix, 42, "test")
        self.assertEqual(sum(1 for j in esc if j["label"] == 0), 20)
        self.assertEqual(rel["te"]["deficit"], 6)
        self.assertEqual(rel["background"]["usados"], 6 + 6)
        self.assertEqual(sum(1 for j in esc if j["source"] == "markov"), 7)
        # test_strict: balanco recalculado so com os elegiveis do strict
        strict = [j for j in js if j["wid"] not in {f"l1:c:{i}" for i in range(10)}]
        esc_s, _ = P.balancear(strict, mix, 42, "test_strict")
        self.assertEqual(sum(1 for j in esc_s if j["label"] == 0), 10)
        self.assertEqual(esc, P.balancear(js, mix, 42, "test")[0])  # deterministico


class TestMarkov(unittest.TestCase):
    def test_gera_casando_gc(self):
        import numpy as np

        rng = random.Random(3)
        seqs = [seq_aleatoria(W, rng, gc=0.40) for _ in range(50)]
        modelo = P.treinar_markov(seqs, 3, W)
        self.assertEqual(int(modelo[0].sum()), 50 * (W - 3))
        hist = P.histograma_gc([P.conta_gc(s) for s in seqs], W)
        out, deficit, _ = P.gerar_markov(modelo, 3, W, 40, hist, np.random.default_rng(7))
        self.assertTrue(all(len(s) == W and P.so_acgt(s) for s in out))
        self.assertEqual(len(out) + deficit, sum(round(40 * h) for h in hist))
        out2, _, _ = P.gerar_markov(modelo, 3, W, 40, hist, np.random.default_rng(7))
        self.assertEqual(out, out2)


@unittest.skipUnless(os.access(MMSEQS, os.X_OK), f"MMseqs2 nao encontrado em {MMSEQS}")
class TestPontaAPonta(unittest.TestCase):
    """Genoma sintetico chr21 (holdout), chr10 (dev) e chr22 (train), com um L1 copiado entre os tres."""

    @classmethod
    def setUpClass(cls):
        import pysam

        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = cls.tmp.name
        rng = random.Random(11)
        chr21, chr22 = list(seq_aleatoria(120000, rng)), list(seq_aleatoria(200000, rng))
        familia, unico21, unico22 = (seq_aleatoria(6000, rng, 0.40) for _ in range(3))
        chr10, unico10 = list(seq_aleatoria(120000, rng)), seq_aleatoria(6000, rng, 0.40)
        l1farm, rmsk = [], []

        def inserir(nome, crom, pos, molde, strand, ident):
            crom[pos:pos + len(molde)] = list(molde)
            for a, b, reg in ((0, 900, "5UTR"), (900, 2000, "ORF1"), (2000, 5800, "ORF2"), (5800, 6000, "3UTR")):
                l1farm.append(linha_l1farm(nome, pos + a, pos + b, "L1PA2", reg, strand, ident))
            rmsk.append(linha_rmsk(nome, pos, pos + len(molde), strand, "L1PA2", "LINE", "L1"))

        inserir("chr21", chr21, 10000, familia, "+", 0.99)
        inserir("chr21", chr21, 50000, unico21, "-", 0.97)
        inserir("chr22", chr22, 10000, mutar(familia, 0.02, rng), "+", 0.99)
        inserir("chr22", chr22, 60000, mutar(familia, 0.02, rng), "+", 0.99)
        inserir("chr22", chr22, 120000, unico22, "+", 0.96)
        inserir("chr10", chr10, 10000, mutar(familia, 0.02, rng), "-", 0.99)
        inserir("chr10", chr10, 50000, unico10, "+", 0.98)
        rmsk.append(linha_rmsk("chr22", 150000, 151000, "+", "L1MA9", "LINE", "L1"))  # L1 antigo
        rmsk.append(linha_rmsk("chr22", 130000, 131000, "+", "MER41", "LTR", "ERV1"))
        for crom, nome, ini, fim in (("chr21", "chr21", 70000, 115000), ("chr22", "chr22", 70000, 115000),
                                     ("chr10", "chr10", 70000, 115000)):
            for i, pos in enumerate(range(ini, fim, 2500)):
                classe, fam = (("SINE", "Alu"), ("SINE", "MIR"), ("DNA", "TcMar-Tigger"), ("LINE", "L2"))[i % 4]
                rmsk.append(linha_rmsk(nome, pos, pos + 300, "+-"[i % 2], fam + "x", classe, fam))
        rmsk.append(linha_rmsk("chr22", 160000, 160050, "+", "(CA)n", "Simple_repeat", "Simple_repeat"))
        s21, s22 = "".join(chr21), "".join(chr22)
        cls.hg38 = os.path.join(d, "hg38.fa")
        escrever_fasta(cls.hg38, [("chr10", "".join(chr10)), ("chr21", s21[:30000].lower() + s21[30000:]),
                                  ("chr22", s22),
                                  ("chr22_KI270731v1_random", seq_aleatoria(5000, rng))])
        pysam.faidx(cls.hg38)
        cls.rmsk = os.path.join(d, "rmsk.txt.gz")
        with gzip.open(cls.rmsk, "wt") as f:
            f.writelines(rmsk)
        cls.l1farm = os.path.join(d, "l1farm.tsv")
        with open(cls.l1farm, "w") as f:
            f.write("## L1Farm\n# teste\n\n\n")
            f.writelines(l1farm)
        genomas = [(f"NC_00000{i}.1 |Fake virus {i}, complete genome|x", seq_aleatoria(5000, rng, 0.45))
                   for i in range(1, 4)]
        cls.retro = os.path.join(d, "retro.fa")
        escrever_fasta(cls.retro, genomas)
        cls.cds = os.path.join(d, "cds.fa")
        escrever_fasta(cls.cds, [("NC_000001.1:100..3200 |pol protein [Fake]|y", genomas[0][1][100:3200]),
                                 ("NC_000001.1:3200..4000 |env protein [Fake]|y", genomas[0][1][3200:4000])])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rodar(self, out, *extra):
        argv = ["--out-dir", out, "--hg38", self.hg38, "--rmsk", self.rmsk, "--l1farm", self.l1farm,
                "--retro-genomes", self.retro, "--retro-cds", self.cds, "--mmseqs", MMSEQS,
                "--tmp-dir", os.path.join(self.d, "tmp"), "--threads", "1", "--limit-chroms", "chr10,chr21,chr22",
                *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main(argv)
        return buf.getvalue()

    def ler(self, out):
        linhas = {}
        for nome in P.CONJUNTOS:
            with open(os.path.join(out, f"{nome}.csv"), newline="") as f:
                leitor = csv.DictReader(f)
                self.assertEqual(tuple(leitor.fieldnames), P.COLUNAS)
                linhas[nome] = list(leitor)
        return linhas

    def test_filter_test_invariantes_e_determinismo(self):
        out1, out2 = os.path.join(self.d, "o1"), os.path.join(self.d, "o2")
        self.rodar(out1)
        self.rodar(out2)
        for nome in P.CONJUNTOS:
            h = [P.sha256_arquivo(os.path.join(o, f"{nome}.csv")) for o in (out1, out2)]
            self.assertEqual(h[0], h[1], nome)
        self.assertEqual(os.listdir(os.path.join(self.d, "tmp")), [])  # temporarios do MMseqs apagados

        linhas = self.ler(out1)
        import json
        with open(os.path.join(out1, "data_meta.json")) as f:
            meta = json.load(f)
        grupos = {n: {r["group"] for r in rs} for n, rs in linhas.items()}
        self.assertTrue(grupos["train"] & grupos["dev"])               # esperado no modo chrom (filter-test)
        self.assertFalse(grupos["dev_strict"] & grupos["train"])
        self.assertFalse(grupos["test_strict"] & (grupos["train"] | grupos["dev"]))
        for nome, rs in linhas.items():
            self.assertTrue(all(len(r["sequence"]) == W and P.so_acgt(r["sequence"]) for r in rs))
            self.assertTrue(all(r["label"] in ("0", "1") for r in rs))
            chroms = {r["chrom"] for r in rs if r["chrom"]}
            self.assertEqual(chroms, {"chr21"} if nome.startswith("test") else
                             {"chr10"} if nome.startswith("dev") else {"chr22"}, nome)
            ids = [r["max_id_train"] for r in rs] + [r["max_id_train_local"] for r in rs]
            if nome == "train":
                self.assertEqual(set(ids), {""})
            else:
                self.assertTrue(all(0 <= float(x) <= 1 for x in ids))
            ori = defaultdict(list)
            for r in rs:
                ori[r["window_id"]].append(r["orientation"])
                if r["orientation"] == "fwd":
                    fwd = r["sequence"]
                else:
                    self.assertEqual(r["sequence"], P.revcomp(fwd))
            self.assertTrue(all(sorted(o) == ["fwd", "rc"] for o in ori.values()))
            # negativos == positivos, exceto o deficit que o balanco registrou (nunca silencioso)
            falta = 2 * meta["balanco"][nome]["background"]["deficit"]
            self.assertEqual(sum(r["label"] == "1" for r in rs) - sum(r["label"] == "0" for r in rs), falta, nome)
            self.assertTrue(all(r["chrom"] == "" for r in rs if r["source"] in ("markov", "retrovirus")))
        # o L1 copiado de chr21 para chr22 sai do test_strict; o unico de chr21 fica
        l1_test = {r["window_id"] for r in linhas["test"] if r["source"] == "l1"}
        l1_strict = {r["window_id"] for r in linhas["test_strict"] if r["source"] == "l1"}
        self.assertTrue(l1_strict and l1_strict < l1_test)
        self.assertTrue(all(10000 <= int(w.split(":")[2].split("-")[0]) < 16000 for w in l1_test - l1_strict))
        # max_id_train: a copia da familia (2% de mutacao) tem identidade alta com o train; o L1 unico, nao
        ids = {r["window_id"]: float(r["max_id_train"]) for r in linhas["test"] if r["source"] == "l1"}
        self.assertTrue(all(ids[w] >= 0.95 for w in l1_test - l1_strict))
        self.assertTrue(all(ids[w] < 0.80 for w in l1_strict))
        # dev: copia da familia fora do dev_strict, L1 unico dentro
        l1_dev = {r["window_id"] for r in linhas["dev"] if r["source"] == "l1"}
        l1_dev_strict = {r["window_id"] for r in linhas["dev_strict"] if r["source"] == "l1"}
        self.assertTrue(l1_dev_strict and l1_dev_strict < l1_dev)
        self.assertTrue(all(50000 <= int(w.split(":")[2].split("-")[0]) < 56000 for w in l1_dev_strict))
        # zona cinza: nada de negativo tocando o L1 antigo do rmsk
        for r in linhas["train"] + linhas["dev"]:
            if r["label"] == "0" and r["chrom"] == "chr22":
                self.assertFalse(int(r["start"]) < 151000 and int(r["end"]) > 150000)
                self.assertFalse(int(r["start"]) < 131000 and int(r["end"]) > 130000)  # LTR

        chk = dict(meta["checagens"])
        self.assertGreater(chk.pop("grupo_em_test_e_train_dev"), 0)  # esperado em filter-test
        self.assertGreater(chk.pop("grupo_em_train_e_dev"), 0)        # esperado no modo chrom
        self.assertTrue(all(v == 0 for v in chk.values()), chk)
        self.assertIn("L1PA2", meta["vazamento"]["l1_test_fora_do_test_strict_por_subfamilia"])
        self.assertIn("easy-linclust", meta["mmseqs_comando"])
        self.assertIn("easy-search", meta["max_id_train"]["comando"])
        faixas = meta["max_id_train"]["faixas"]["test"]["max_id_train"]["l1"]
        self.assertEqual(sum(faixas.values()), len(l1_test))
        vs = meta["max_id_train"]["l1_vs_strict"]["test"]
        self.assertEqual(vs["l1_test_strict"], len(l1_strict))

        saida = self.rodar(out1, "--only-if-changed")
        self.assertIn("nada a fazer", saida)

    def test_purge_train_e_pol(self):
        out = os.path.join(self.d, "purge")
        self.rodar(out, "--leak-policy", "purge-train", "--retrovirus-mode", "pol", "--dev-mode", "hash")
        linhas = self.ler(out)
        self.assertEqual(linhas["test"], linhas["test_strict"])
        self.assertEqual(linhas["dev"], linhas["dev_strict"])
        grupos = {n: {r["group"] for r in rs} for n, rs in linhas.items()}
        self.assertFalse(grupos["test"] & (grupos["train"] | grupos["dev"]))
        self.assertFalse(grupos["train"] & grupos["dev"])
        retro = {r["detail"] for rs in linhas.values() for r in rs if r["source"] == "retrovirus"}
        self.assertLessEqual(retro, {"NC_000001.1"})
        import json
        with open(os.path.join(out, "data_meta.json")) as f:
            meta = json.load(f)
        self.assertGreater(meta["vazamento"]["purge_por_fonte"].get("l1", 0), 0)
        self.assertEqual(meta["descartes"]["retro_registros_usados"], 1)


if __name__ == "__main__":
    unittest.main()
