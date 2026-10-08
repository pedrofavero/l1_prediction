#!/usr/bin/env python
"""Extrai embeddings congelados dos CSVs L1: um .npy float16 por split.

Saida em --out-dir (padrao $L1_EMB_DIR/<model_tag>/<ident>/):
  <split>.npy        float16 [n_linhas, dim], na ordem das linhas do CSV
  <split>.index.csv  window_id,orientation,label,source,max_id_train,max_id_train_local
  meta.json          modelo, revision, origem dos pesos, dim, camada, pooling e, por
                     split, sha256 do CSV/.npy/index, tempo, seq/s e VRAM pico
Nada compactado (o storage do CISIA bloqueia): np.save sem compressao, CSV/JSON
puros, nunca torch.save. Antes de tudo os CSVs sao conferidos contra o
data_meta.json (--verify-only faz so isso). Um split ja extraido com o mesmo
modelo, revision, adaptador e sha256 do CSV, e com o .npy intacto, e pulado.

Adaptadores (familia de modelo) em ADAPTADORES; cada um define tokenizacao,
forward e pooling (`carregar()` e `embed(seqs) -> float32 [n, dim]`):
  nt2   Nucleotide Transformer v2 (AutoModelForMaskedLM, hidden_states[-1],
        media sobre tokens validos)
  fake  contagem de 3-mers (64 dims), sem torch nem GPU: so para os testes

Device: --device auto|cuda|mps|cpu (auto = cuda > mps > cpu; pedido indisponivel
aborta). dtype: --dtype auto|bfloat16|float32 (auto = bf16 com autocast em cuda, e
em mps se suportado; float32 em cpu; fp16 nunca, por risco de overflow). Antes de
extrair, PARIDADE: 64 linhas do dev pelo device/dtype escolhido e pela CPU em
float32; aborta se o cosseno minimo for < 0,999.
Para um modelo novo (nt3, dnabert2, evo2): subclasse de Adaptador + entrada em
ADAPTADORES; o resto do pipeline nao muda.

Requer PYTHONPATH="$REPO_ROOT/scripts" (lib.l1_dados); o .sbatch exporta.
Uso: extract_embeddings.py --model-tag nt2_50m --family nt2 --model-id ID --revision SHA
                           --ident ident98 [--data-dir D] [--out-dir D] [--splits ...]
                           [--device auto] [--dtype auto] [--batch N] [--weights-note TXT]
     extract_embeddings.py --verify-only --data-dir D
"""
import argparse
import contextlib
import csv
import json
import os
import platform
import subprocess
import sys
import time

try:
    from lib.l1_dados import CONJUNTOS, COLUNAS_INDICE, IntegridadeErro, ler_csv_l1, sha256_arquivo, verificar_integridade
except ModuleNotFoundError as e:
    if e.name not in ("lib", "lib.l1_dados"):
        raise
    sys.exit('ERRO: modulo lib.l1_dados nao encontrado (PYTHONPATH sem <repo>/scripts). Antes de rodar:\n'
             '    export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"')

MAX_LEN_TOKENS = 2048  # contexto maximo do NT v2
LOTES_DETALHADOS = 3   # lotes iniciais com VRAM e throughput no log
DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "bfloat16", "float32")
BATCH_PADRAO = {"cuda": 64, "mps": 32, "cpu": 16}  # mps conservador para 18 GB de memoria unificada
PARIDADE_N = 64
PARIDADE_MIN = 0.999


def resolver_device(pedido, cuda_ok, mps_ok):
    """auto -> cuda > mps > cpu. Device pedido explicitamente e indisponivel -> RuntimeError."""
    if pedido == "auto":
        return "cuda" if cuda_ok else "mps" if mps_ok else "cpu"
    if pedido == "cuda" and not cuda_ok:
        raise RuntimeError("--device cuda, mas torch.cuda.is_available() == False: nenhuma GPU visivel. Na "
                           "particao shared o recurso e --gres=mps:<pct>, NAO --gres=gpu:")
    if pedido == "mps" and not mps_ok:
        raise RuntimeError("--device mps, mas torch.backends.mps.is_available() == False")
    return pedido


def resolver_dtype(pedido, device, bf16_ok):
    """auto -> bfloat16 em cuda/mps se suportado, senao float32; float32 em cpu. Nunca float16."""
    if pedido == "float32":
        return "float32"
    if pedido == "bfloat16":
        if not bf16_ok:
            raise RuntimeError(f"--dtype bfloat16, mas bf16 nao e suportado em {device}")
        return "bfloat16"
    return "bfloat16" if device != "cpu" and bf16_ok else "float32"


def cosseno(a, b):
    """Cosseno linha a linha entre duas matrizes -> (minimo, media)."""
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide((a * b).sum(1), den, out=np.zeros(len(a)), where=den > 0)
    return float(cos.min()), float(cos.mean())


def preparar_torch(pedido_device, pedido_dtype):
    """Importa torch (com o fallback do MPS ligado antes) e resolve device/dtype -> (device, dtype, info)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    device = resolver_device(pedido_device, torch.cuda.is_available(), mps_ok)
    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        chip = torch.cuda.get_device_name(0)
    elif device == "mps":
        try:
            a = torch.randn(8, 8, device="mps")
            with torch.autocast("mps", dtype=torch.bfloat16):
                bf16_ok = (a @ a).dtype == torch.bfloat16
        except Exception:  # noqa: BLE001 (autocast/bf16 no MPS varia com a versao do torch e do macOS)
            bf16_ok = False
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        chip = r.stdout.strip() or platform.machine()
    else:
        bf16_ok = False
        chip = platform.processor() or platform.machine()
    dtype = resolver_dtype(pedido_dtype, device, bf16_ok)
    return device, dtype, {"device": device, "dtype_forward": dtype + (" (autocast)" if dtype == "bfloat16" else ""),
                           "bf16_suportado": bf16_ok, "chip": chip, "torch": torch.__version__,
                           "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") if device == "mps" else None}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Adaptador:
    """Interface por familia de modelo. carregar() -> dict de info; embed(seqs) -> np.float32 [n, dim]."""

    familia = None
    versao = 1       # incrementar quando tokenizacao/forward/pooling mudar (invalida o reaproveitamento)
    camada = None
    pooling = None
    usa_torch = True

    def __init__(self, model_id, revision, device="cpu", dtype="float32"):
        self.model_id, self.revision = model_id, revision
        self.device, self.dtype = device, dtype
        self.dim = None

    def carregar(self):
        raise NotImplementedError

    def embed(self, seqs):
        raise NotImplementedError

    def vram_pico_gb(self):
        return None

    def info_lote(self):
        """Detalhe do ultimo lote para o log (ex.: tokens por sequencia)."""
        return ""


class AdaptadorNT2(Adaptador):
    familia = "nt2"
    camada = "hidden_states[-1] (ultima camada)"
    pooling = "media sobre tokens com attention_mask=1, excluindo CLS/EOS/BOS/PAD/MASK"

    def carregar(self):
        import torch
        import transformers
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.dev = torch.device("cuda:0" if self.device == "cuda" else self.device)
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id, revision=self.revision,
                                                          trust_remote_code=True).to(self.dev).eval()
        ids = [getattr(self.tok, f"{n}_token_id", None) for n in ("cls", "eos", "bos", "pad", "mask")]
        self.especiais = torch.tensor(sorted({i for i in ids if i is not None}), device=self.dev)
        self.dim = int(self.model.config.hidden_size)
        self._tokens = None
        self._mem_max = 0
        info = {"tokens_especiais_excluidos": self.especiais.tolist(),
                "n_parametros": sum(p.numel() for p in self.model.parameters()),
                "versoes": {"torch": torch.__version__, "transformers": transformers.__version__}}
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            torch.cuda.reset_peak_memory_stats()
            info.update(gpu=props.name, vram_total_gb=round(props.total_memory / 2**30, 1),
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        return info

    def embed(self, seqs):
        torch = self.torch
        enc = self.tok(list(seqs), padding="longest", truncation=True, max_length=MAX_LEN_TOKENS,
                       return_tensors="pt")
        ids = enc["input_ids"].to(self.dev)
        mask = enc["attention_mask"].to(self.dev)
        amp = (torch.autocast(self.device, dtype=torch.bfloat16) if self.dtype == "bfloat16"
               else contextlib.nullcontext())
        with torch.no_grad(), amp:
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        h = out.hidden_states[-1].float()
        valido = (mask.bool() & ~torch.isin(ids, self.especiais)).unsqueeze(-1).float()
        n = valido.sum(1)
        if (n == 0).any():
            raise RuntimeError("sequencia sem nenhum token valido para o pooling")
        emb = (h * valido).sum(1) / n
        if not torch.isfinite(emb).all():
            raise RuntimeError("embedding nao finito (NaN/inf) no forward")
        self._tokens = (int(mask.sum(1).min()), int(mask.sum(1).max()), int(n.min()), int(n.max()))
        emb = emb.cpu().numpy()
        if self.device == "mps":  # o MPS nao tem contador de pico: maximo amostrado apos cada lote
            self._mem_max = max(self._mem_max, torch.mps.driver_allocated_memory())
        return emb

    def vram_pico_gb(self):
        """Pico de VRAM (cuda) ou maximo amostrado da memoria do driver (mps); None na cpu."""
        if self.device == "cuda":
            return round(self.torch.cuda.max_memory_allocated() / 2**30, 2)
        if self.device == "mps":
            return round(self._mem_max / 2**30, 2)
        return None

    def info_lote(self):
        if self._tokens is None:
            return ""
        t = self._tokens
        return f"tokens/seq {t[0]}-{t[1]} (validos no pooling {t[2]}-{t[3]})"


class AdaptadorFalso(Adaptador):
    """Embedding deterministico a partir da sequencia (3-mers normalizados). So para testes."""

    familia = "fake"
    camada = "-"
    pooling = "contagem de 3-mers / numero de 3-mers"
    usa_torch = False

    def carregar(self):
        import numpy as np

        self.np = np
        self.cod = np.full(256, 0, dtype=np.int64)
        for i, b in enumerate(b"ACGT"):
            self.cod[b] = i
        self.dim = 64
        return {"dispositivo": "cpu"}

    def embed(self, seqs):
        np = self.np
        out = np.zeros((len(seqs), 64), dtype=np.float32)
        for i, s in enumerate(seqs):
            a = self.cod[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
            k = a[:-2] * 16 + a[1:-1] * 4 + a[2:]
            out[i] = np.bincount(k, minlength=64) / max(len(k), 1)
        return out


# Pendentes: "nt3" (NTv3_650M_post, config.sh), "dnabert2" (env proprio, BertConfig explicito),
# "evo2" (env proprio, Hopper/FP8). Registrar aqui quando implementados.
ADAPTADORES = {"nt2": AdaptadorNT2, "fake": AdaptadorFalso}


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


def gravar_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def chave_modelo(args, adaptador_cls, device, dtype):
    """Reaproveitamento so com a mesma chave: embeddings de devices/dtypes diferentes nao se misturam."""
    return {"model_tag": args.model_tag, "familia": args.family, "model_id": args.model_id,
            "revision": args.revision, "versao_adaptador": adaptador_cls.versao, "device": device,
            "dtype_forward": dtype, "dtype_saida": "float16"}


def embed_em_lotes(adaptador, seqs, batch):
    import numpy as np

    return np.concatenate([adaptador.embed(seqs[i:i + batch]) for i in range(0, len(seqs), batch)])


def checar_paridade(cls, args, adaptador, data_dir, batch):
    """PARIDADE_N linhas do dev (espacadas no CSV) no device/dtype escolhido vs CPU float32 -> resultado.

    Nao aplicavel quando o proprio device ja e a referencia (cpu/float32) ou o adaptador nao usa torch.
    """
    if not cls.usa_torch or (adaptador.device == "cpu" and adaptador.dtype == "float32"):
        return {"aplicavel": False, "motivo": "device ja e a referencia (cpu float32) ou adaptador sem torch"}
    import numpy as np

    seqs_dev = ler_csv_l1(os.path.join(data_dir, "dev.csv"), com_sequencia=True)["sequence"]
    idx = sorted(set(np.linspace(0, len(seqs_dev) - 1, PARIDADE_N).round().astype(int).tolist()))
    seqs = [seqs_dev[i] for i in idx]
    a = embed_em_lotes(adaptador, seqs, batch)
    ref = cls(args.model_id, args.revision, "cpu", "float32")
    ref.carregar()
    b = embed_em_lotes(ref, seqs, BATCH_PADRAO["cpu"])
    del ref
    cmin, cmed = cosseno(a, b)
    return {"aplicavel": True, "n": len(seqs), "linhas_do_dev": "espacadas uniformemente no CSV",
            "referencia": "cpu float32", "device": adaptador.device, "dtype": adaptador.dtype,
            "cos_min": round(cmin, 6), "cos_media": round(cmed, 6), "limite": PARIDADE_MIN,
            "aprovado": cmin >= PARIDADE_MIN}


def reaproveitavel(out_dir, split, entrada, sha_csv):
    if not entrada or entrada.get("sha256_csv") != sha_csv:
        return False
    npy = os.path.join(out_dir, f"{split}.npy")
    idx = os.path.join(out_dir, f"{split}.index.csv")
    return (os.path.exists(npy) and os.path.exists(idx) and sha256_arquivo(npy) == entrada.get("sha256_npy")
            and sha256_arquivo(idx) == entrada.get("sha256_index"))


def extrair_split(adaptador, csv_path, out_dir, split, batch):
    """Extrai um split para <split>.npy (via .tmp + os.replace) e <split>.index.csv. -> entrada do meta."""
    import numpy as np

    dados = ler_csv_l1(csv_path, com_sequencia=True)
    seqs = dados["sequence"]
    n = len(seqs)
    final = os.path.join(out_dir, f"{split}.npy")
    tmp = final + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, adaptador.dim))
    t0 = time.time()
    proximo_log = 0.1
    for k, i in enumerate(range(0, n, batch)):
        emb = adaptador.embed(seqs[i:i + batch])
        if emb.shape != (len(seqs[i:i + batch]), adaptador.dim):
            raise RuntimeError(f"embedding com shape {emb.shape}, esperado ({len(seqs[i:i + batch])}, {adaptador.dim})")
        e16 = emb.astype(np.float16)
        if not np.isfinite(e16).all():
            raise RuntimeError("embedding estoura o float16 (|x| > 65504)")
        arr[i:i + len(e16)] = e16
        feitos = i + len(e16)
        if k < LOTES_DETALHADOS or feitos / n >= proximo_log or feitos == n:
            dt = time.time() - t0
            log(f"{split}: {feitos}/{n} ({100 * feitos / n:.0f}%), {feitos / dt if dt else 0:.1f} seq/s, "
                f"VRAM pico {adaptador.vram_pico_gb()} GB {adaptador.info_lote()}")
            while feitos / n >= proximo_log:
                proximo_log += 0.1
    tempo = time.time() - t0
    arr.flush()
    del arr
    os.replace(tmp, final)
    idx_path = os.path.join(out_dir, f"{split}.index.csv")
    with open(idx_path + ".tmp", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(COLUNAS_INDICE)
        for linha in zip(*(dados[c] for c in COLUNAS_INDICE)):
            wr.writerow(["" if v is None else v for v in linha])
    os.replace(idx_path + ".tmp", idx_path)
    return {"n": n, "sha256_npy": sha256_arquivo(final), "sha256_index": sha256_arquivo(idx_path),
            "tempo_s": round(tempo, 1), "seq_por_s": round(n / tempo, 1) if tempo else None,
            "vram_pico_gb": adaptador.vram_pico_gb()}


def main(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-tag")
    ap.add_argument("--family", choices=sorted(ADAPTADORES))
    ap.add_argument("--model-id")
    ap.add_argument("--revision")
    ap.add_argument("--ident", help="ident95 | ident98 (define os caminhos padrao)")
    ap.add_argument("--data-dir", help="padrao: $L1_DATASET_DIR/<ident>")
    ap.add_argument("--out-dir", help="padrao: $L1_EMB_DIR/<model_tag>/<ident>")
    ap.add_argument("--splits", nargs="+", default=list(CONJUNTOS), choices=CONJUNTOS)
    ap.add_argument("--device", choices=DEVICES, default="auto", help="auto = cuda > mps > cpu")
    ap.add_argument("--dtype", choices=DTYPES, default="auto",
                    help="auto = bfloat16 (autocast) em cuda/mps se suportado, float32 em cpu; nunca float16")
    ap.add_argument("--batch", type=int, help=f"padrao por device: {BATCH_PADRAO}")
    ap.add_argument("--weights-note", default="", help="origem dos pesos, gravada no meta.json")
    ap.add_argument("--verify-only", action="store_true", help="so confere os sha256 dos CSVs e sai")
    args = ap.parse_args(argv)

    data_dir = args.data_dir or (env("L1_DATASET_DIR") and args.ident and os.path.join(env("L1_DATASET_DIR"), args.ident))
    if not data_dir:
        sys.exit("ERRO: passe --data-dir ou --ident com L1_DATASET_DIR definido (environments/config.sh)")
    try:
        hashes = verificar_integridade(data_dir)
    except IntegridadeErro as e:
        sys.exit(f"ERRO de INTEGRIDADE: {e}")
    log(f"integridade OK: {', '.join(f'{k} {v[:12]}' for k, v in hashes.items())}")
    if args.verify_only:
        return

    faltando = [n for n in ("model_tag", "family", "model_id", "revision", "ident") if not getattr(args, n)]
    if faltando:
        sys.exit(f"ERRO: faltam {', '.join('--' + n.replace('_', '-') for n in faltando)}")
    out_dir = args.out_dir or (env("L1_EMB_DIR") and os.path.join(env("L1_EMB_DIR"), args.model_tag, args.ident))
    if not out_dir:
        sys.exit("ERRO: passe --out-dir ou defina L1_EMB_DIR (environments/config.sh)")
    os.makedirs(out_dir, exist_ok=True)
    cls = ADAPTADORES[args.family]
    if cls.usa_torch:
        try:
            device, dtype, info_device = preparar_torch(args.device, args.dtype)
        except RuntimeError as e:
            sys.exit(f"ERRO: {e}")
    else:
        device, dtype, info_device = "cpu", "float32", {"device": "cpu", "dtype_forward": "float32"}
    batch = args.batch or BATCH_PADRAO[device]
    log(f"device {device}, dtype {dtype}, batch {batch} ({info_device})")
    chave = chave_modelo(args, cls, device, dtype)
    meta_path = os.path.join(out_dir, "meta.json")
    try:
        with open(meta_path) as f:
            antigo = json.load(f)
    except (OSError, ValueError):
        antigo = {}
    splits_antigos = antigo.get("splits", {}) if antigo.get("chave") == chave else {}

    pendentes = [s for s in args.splits if not reaproveitavel(out_dir, s, splits_antigos.get(s), hashes[s])]
    for s in args.splits:
        if s not in pendentes:
            log(f"{s}: reaproveitado (mesmo modelo, revision, adaptador e sha256 do CSV; .npy intacto)")
    import numpy

    meta = {"chave": chave, **chave, "origem_pesos": args.weights_note or None, "ident": args.ident,
            "data_dir": data_dir, "camada": cls.camada, "pooling": cls.pooling, "batch": batch,
            "git": git_commit(), "hostname": platform.node(), "slurm_job_id": env("SLURM_JOB_ID"),
            "versoes": {"python": platform.python_version(), "numpy": numpy.__version__},
            "splits": {s: e for s, e in splits_antigos.items() if s in args.splits and s not in pendentes}}
    if pendentes:
        adaptador = cls(args.model_id, args.revision, device, dtype)
        t0 = time.time()
        info = {**info_device, **adaptador.carregar()}
        log(f"modelo {args.model_id}@{args.revision[:12]} ({args.family}) carregado em {time.time() - t0:.1f} s; "
            f"dim {adaptador.dim}; {info}")
        paridade = checar_paridade(cls, args, adaptador, data_dir, batch)
        log(f"paridade: {paridade}")
        if paridade.get("aplicavel") and not paridade["aprovado"]:
            sys.exit(f"ERRO de PARIDADE: cosseno minimo {paridade['cos_min']} < {PARIDADE_MIN} entre {device}/{dtype} "
                     "e cpu/float32; nada foi extraido. Tente --dtype float32.")
        meta.update(dim=adaptador.dim, ambiente=info, paridade=paridade)
        for s in pendentes:
            log(f"{s}: extraindo")
            entrada = extrair_split(adaptador, os.path.join(data_dir, f"{s}.csv"), out_dir, s, batch)
            entrada["sha256_csv"] = hashes[s]
            meta["splits"][s] = entrada
            gravar_json(meta_path, meta)  # grava a cada split: um timeout nao perde o que ja foi feito
    else:
        meta.update(dim=antigo.get("dim"), ambiente=antigo.get("ambiente"), paridade=antigo.get("paridade"))
    meta["gerado_em"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    gravar_json(meta_path, meta)
    log(f"embeddings em {out_dir}: " + ", ".join(f"{s} {meta['splits'][s]['n']}" for s in args.splits))


if __name__ == "__main__":
    main()
