"""Testes do baseline L1 (metrics, integridade, extracao com adaptador falso + probe).

Rodar no Mac: ~/l1_data/venv/bin/python -m unittest discover scripts/train/l1/tests
Nao precisa de GPU, torch nem modelo: o ponta a ponta usa --family fake.
"""
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
L1_DIR = os.path.dirname(AQUI)
SCRIPTS = os.path.dirname(os.path.dirname(L1_DIR))


def carregar_modulo(nome, path):
    spec = importlib.util.spec_from_file_location(nome, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = carregar_modulo("metrics", os.path.join(L1_DIR, "metrics.py"))
D = carregar_modulo("l1_dados", os.path.join(SCRIPTS, "lib", "l1_dados.py"))
COLUNAS = ("sequence", "label", "source", "detail", "window_id", "orientation", "chrom", "start", "end",
           "strand", "group", "gc", "max_id_train", "max_id_train_local")
CONJUNTOS = ("train", "dev", "dev_strict", "test", "test_strict")


def ler_bytes(path):
    with open(path, "rb") as f:
        return bytearray(f.read())


def gravar_bytes(path, dados):
    with open(path, "wb") as f:
        f.write(bytes(dados))


def conj(window_id, label, source, max_id=None, max_id_local=None, orientation=None):
    n = len(label)
    return {"window_id": np.array(window_id), "orientation": np.array(orientation or ["fwd"] * n),
            "label": np.array(label, dtype=int), "source": np.array(source),
            "max_id_train": np.array(max_id if max_id is not None else [None] * n, dtype=float),
            "max_id_train_local": np.array(max_id_local if max_id_local is not None else [None] * n, dtype=float)}


class TestMetricas(unittest.TestCase):
    def test_por_janela_media_fwd_rc(self):
        c = conj(["a", "a", "b", "b"], [1, 1, 0, 0], ["l1", "l1", "te", "te"], orientation=["fwd", "rc"] * 2)
        jan, p = M.por_janela(c, [0.2, 0.6, 0.9, 0.1])
        self.assertEqual(list(jan["window_id"]), ["a", "b"])
        np.testing.assert_allclose(p, [0.4, 0.5])
        self.assertNotIn("orientation", jan)
        c["label"][1] = 0
        with self.assertRaises(ValueError):
            M.por_janela(c, [0.2, 0.6, 0.9, 0.1])

    def test_faixas_de_identidade(self):
        ids = [0.79, 0.80, 0.8999, 0.90, 0.95, 1.0, 0.99]
        c = conj([str(i) for i in range(7)], [1, 1, 1, 1, 1, 1, 0], ["l1"] * 6 + ["te"], max_id=ids)
        pred = np.array([1, 0, 1, 1, 0, 1, 1])
        f = M.recall_por_faixa(c["max_id_train"], c["source"], c["label"], pred)
        self.assertEqual({k: (v["n"], v["recall"]) for k, v in f.items()},
                         {"<0.80": (1, 1.0), "0.80-0.90": (2, 0.5), "0.90-0.95": (1, 1.0), ">=0.95": (2, 0.5)})
        vazio = M.recall_por_faixa(np.array([np.nan]), np.array(["l1"]), np.array([1]), np.array([1]))
        self.assertEqual(vazio["<0.80"], {"n": 0, "recall": None})
        self.assertEqual(vazio["sem_valor"], {"n": 1})

    def test_limiar_do_dev_aplicado_no_test(self):
        t, mcc = M.limiar_mcc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(t, 0.35)                 # empate de MCC com 0.8: fica o mais perto de 0,5
        self.assertAlmostEqual(mcc, 2 / 12 ** 0.5)
        test = M.binarias([0, 1, 1, 0], [0.30, 0.36, 0.9, 0.2], t)
        self.assertEqual(test["limiar"], 0.35)
        self.assertEqual(test["accuracy"], 1.0)   # com 0,5 o 0,36 seria falso negativo
        self.assertEqual(M.binarias([0, 1, 1, 0], [0.30, 0.36, 0.9, 0.2], 0.5)["accuracy"], 0.75)

    def test_limiares_por_nivel(self):
        c = conj(["a", "a", "b", "b"], [1, 1, 0, 0], ["l1", "l1", "te", "te"], orientation=["fwd", "rc"] * 2)
        lim = M.limiares_do_dev(c, [0.9, 0.3, 0.2, 0.25])
        self.assertEqual(lim["linha"], 0.3)       # separa linhas: 0.3 positivo, 0.25 negativo
        self.assertEqual(lim["janela"], 0.6)      # janelas: a = 0.6, b = 0.225

    def test_por_source_e_alvo_vs_negativos(self):
        src = np.array(["l1", "l1", "retrovirus", "te", "te", "markov", "background"])
        y = np.array([1, 1, 1, 0, 0, 0, 0])
        prob = np.array([0.9, 0.2, 0.1, 0.8, 0.1, 0.0, 0.3])
        ps = M.por_source(src, y, (prob >= 0.5).astype(int))
        self.assertEqual(ps["l1"], {"n": 2, "recall": 0.5})
        self.assertEqual(ps["retrovirus"], {"n": 1, "recall": 0.0})
        self.assertEqual(ps["te"], {"n": 2, "especificidade": 0.5})
        self.assertEqual(ps["markov"], {"n": 1, "especificidade": 1.0})
        l1 = M.alvo_vs_negativos(src, y, prob, "l1", 0.5)
        self.assertEqual((l1["n"], l1["n_pos"]), (6, 2))  # sem o retrovirus
        self.assertIsNone(M.alvo_vs_negativos(src[3:], y[3:], prob[3:], "l1", 0.5))

    def test_auroc_none_com_uma_classe(self):
        b = M.binarias([1, 1], [0.2, 0.9], 0.5)
        self.assertIsNone(b["auroc"])
        self.assertIsNone(b["auprc"])
        self.assertEqual(b["n_pos"], 2)


def escrever_dataset(d, rng, n_janelas=None):
    """Dataset sintetico no contrato real: positivos ricos em A/T, negativos uniformes."""
    n_janelas = n_janelas or {"train": 120, "dev": 40, "dev_strict": 30, "test": 40, "test_strict": 30}
    hashes = {}
    for nome, n in n_janelas.items():
        path = os.path.join(d, f"{nome}.csv")
        with open(path, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(COLUNAS)
            for i in range(n):
                label = i % 2
                src = (("l1", "l1", "retrovirus")[i % 3] if label else ("te", "background", "markov")[i % 3])
                pesos = [0.35, 0.15, 0.15, 0.35] if label else [0.25] * 4
                seq = "".join(rng.choices("ACGT", weights=pesos, k=1024))
                comp = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
                ids = ("", "") if nome == "train" else (f"{rng.random():.4f}", f"{rng.random():.4f}")
                for ori, s in (("fwd", seq), ("rc", comp)):
                    wr.writerow([s, label, src, "", f"{src}:{nome}:{i}", ori, "", "", "", "", f"g{i}", "0.4", *ids])
        hashes[nome] = hashlib.sha256(ler_bytes(path)).hexdigest()
    with open(os.path.join(d, "data_meta.json"), "w") as f:
        json.dump({"sha256": hashes}, f)
    return hashes


class TestIntegridade(unittest.TestCase):
    def test_hash_certo_passa_errado_aborta(self):
        with tempfile.TemporaryDirectory() as d:
            hashes = escrever_dataset(d, random.Random(0), {c: 4 for c in CONJUNTOS})
            self.assertEqual(D.verificar_integridade(d), hashes)
            path = os.path.join(d, "dev.csv")
            dados = ler_bytes(path)
            dados[-3] = ord("A") if dados[-3] != ord("A") else ord("C")
            gravar_bytes(path, dados)
            with self.assertRaises(D.IntegridadeErro) as ctx:
                D.verificar_integridade(d)
            self.assertIn("dev.csv", str(ctx.exception))
            os.remove(os.path.join(d, "test.csv"))
            with self.assertRaises(D.IntegridadeErro):
                D.verificar_integridade(d)


class TestPontaAPonta(unittest.TestCase):
    """CSV sintetico -> extract_embeddings.py --family fake -> train_probe.py --mlp, por subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        cls.data = os.path.join(cls.d, "data")
        cls.emb = os.path.join(cls.d, "emb")
        cls.res = os.path.join(cls.d, "res")
        os.makedirs(cls.data)
        escrever_dataset(cls.data, random.Random(1))
        cls.env = {**os.environ, "PYTHONPATH": SCRIPTS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rodar(self, script, *args, ok=True):
        r = subprocess.run([sys.executable, os.path.join(L1_DIR, script), *args], capture_output=True, text=True,
                           env=self.env)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def extrair(self, emb):
        return self.rodar("extract_embeddings.py", "--model-tag", "fake", "--family", "fake", "--model-id", "fake",
                          "--revision", "0" * 40, "--ident", "synth", "--data-dir", self.data, "--out-dir", emb,
                          "--batch", "16", "--weights-note", "sintetico")

    def probe(self, emb, *extra, ok=True):
        return self.rodar("train_probe.py", "--model-tag", "fake", "--ident", "synth", "--data-dir", self.data,
                          "--emb-dir", emb, "--out-root", self.res, *extra, ok=ok)

    def test_extracao_probe_e_reaproveitamento(self):
        self.rodar("extract_embeddings.py", "--verify-only", "--data-dir", self.data)
        self.extrair(self.emb)
        with open(os.path.join(self.emb, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["origem_pesos"], "sintetico")
        for s in CONJUNTOS:
            X = np.load(os.path.join(self.emb, f"{s}.npy"))
            self.assertEqual(X.dtype, np.float16)
            with open(os.path.join(self.emb, f"{s}.index.csv")) as f:
                idx = list(csv.DictReader(f))
            self.assertEqual(X.shape, (len(idx), 64))
            self.assertEqual(meta["splits"][s]["sha256_csv"], D.sha256_arquivo(os.path.join(self.data, f"{s}.csv")))
        with open(os.path.join(self.emb, "train.index.csv")) as f:
            self.assertTrue(all(r["max_id_train"] == "" for r in csv.DictReader(f)))
        self.assertFalse([n for n in os.listdir(self.emb) if n.endswith(".tmp")])

        segunda = self.extrair(self.emb)
        self.assertEqual(segunda.stdout.count("reaproveitado"), len(CONJUNTOS))

        saida = self.probe(self.emb, "--mlp").stdout
        self.assertIn("recall l1 por faixa", saida)
        (run,) = os.listdir(os.path.join(self.res, "fake", "synth"))
        out = os.path.join(self.res, "fake", "synth", run)
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        self.assertEqual(set(rep["modelos"]), {"logreg", "mlp"})
        lr = rep["modelos"]["logreg"]
        self.assertEqual(len(lr["grade"]), 5)
        self.assertEqual(set(lr["conjuntos"]), {"dev", "dev_strict", "test", "test_strict"})
        a = lr["conjuntos"]["test"]["janela"]["dev"]
        self.assertEqual(set(a), {"total", "por_source", "l1_vs_negativos", "retrovirus_vs_negativos",
                                  "faixas_max_id_train", "faixas_max_id_train_local"})
        self.assertEqual(a["total"]["n"], 40)
        self.assertEqual(lr["conjuntos"]["test"]["linha"]["0.5"]["total"]["n"], 80)
        self.assertGreater(lr["conjuntos"]["test"]["linha"]["0.5"]["total"]["mcc"], 0.8)  # sinal sintetico forte
        self.assertEqual(rep["sha256_csv"], D.verificar_integridade(self.data))
        for prefixo in ("predictions", "predictions_mlp"):
            for s in ("dev", "dev_strict", "test", "test_strict"):
                with open(os.path.join(out, f"{prefixo}_{s}.csv")) as f:
                    linhas = list(csv.DictReader(f))
                self.assertEqual(len(linhas), 2 * {"dev": 40, "dev_strict": 30, "test": 40, "test_strict": 30}[s])
                self.assertEqual(list(linhas[0]), ["window_id", "orientation", "source", "label", "prob",
                                                   "max_id_train", "max_id_train_local"])
        self.assertTrue(os.path.exists(os.path.join(out, "probe_logreg.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "probe_mlp_w0.npy")))
        compactados = [n for n in os.listdir(out) + os.listdir(self.emb)
                       if n.endswith((".gz", ".zip", ".npz", ".pt", ".pth", ".bin"))]
        self.assertEqual(compactados, [])

    def test_probe_aborta_com_npy_corrompido(self):
        emb = os.path.join(self.d, "emb_corrompido")
        self.extrair(emb)
        path = os.path.join(emb, "dev.npy")
        dados = ler_bytes(path)
        dados[-1] ^= 0xFF
        gravar_bytes(path, dados)
        r = self.probe(emb, ok=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dev.npy", r.stderr)
        self.assertIn("sha256", r.stderr)


if __name__ == "__main__":
    unittest.main()
