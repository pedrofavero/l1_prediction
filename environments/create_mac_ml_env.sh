#!/bin/bash
# Cria o venv de ML do Mac (~/l1_data/venv-ml): extracao de embeddings (torch com
# MPS) e probe. SEPARADO do ~/l1_data/venv do data_prep, que nao e tocado.
#
# Uso: environments/create_mac_ml_env.sh [--recreate]
#   --recreate  apaga e recria o venv se ele ja existir (senao, aborta)
#
# Python base por caminho absoluto (ML_BASE_PYTHON, padrao: python.org 3.12);
# todo pip roda como "$ML_PYTHON" -m pip. transformers na mesma faixa do servidor.
# No fim valida imports, MPS e bf16 no MPS, e grava o pip freeze em $ML_VENV/freeze.txt.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.local.sh"

ML_BASE_PYTHON="${ML_BASE_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12}"
RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERRO: argumento desconhecido: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -x "$ML_BASE_PYTHON" ]; then
    echo "ERRO: Python base nao encontrado em $ML_BASE_PYTHON (exporte ML_BASE_PYTHON)" >&2
    exit 1
fi
if [ -e "$ML_VENV" ]; then
    if [ "$RECREATE" -eq 0 ]; then
        echo "ERRO: $ML_VENV ja existe; use --recreate para apagar e recriar" >&2
        exit 1
    fi
    rm -rf "$ML_VENV"
fi

echo "== Python base: $ML_BASE_PYTHON ($("$ML_BASE_PYTHON" --version))"
"$ML_BASE_PYTHON" -m venv "$ML_VENV"
"$ML_PYTHON" -m pip install --upgrade pip
"$ML_PYTHON" -m pip install torch "transformers>=4.55,<5" scikit-learn numpy huggingface_hub safetensors

echo "== Validando"
"$ML_PYTHON" - <<'PY'
import numpy, safetensors, sklearn, torch, transformers, huggingface_hub

print(f"torch {torch.__version__}, transformers {transformers.__version__}, numpy {numpy.__version__}, "
      f"scikit-learn {sklearn.__version__}, huggingface_hub {huggingface_hub.__version__}")
if not torch.backends.mps.is_available():
    raise SystemExit("ERRO: MPS indisponivel neste torch/macOS")
a = torch.randn(64, 64, device="mps")
try:
    with torch.autocast("mps", dtype=torch.bfloat16):
        b = a @ a
    print(f"MPS OK; bf16 com autocast OK ({b.dtype})")
except Exception as e:  # noqa: BLE001
    print(f"MPS OK; bf16 NAO suportado ({type(e).__name__}: {e}); a extracao vai usar float32")
PY
"$ML_PYTHON" -m pip freeze > "$ML_VENV/freeze.txt"
echo "== Pronto: $ML_PYTHON (freeze em $ML_VENV/freeze.txt)"
