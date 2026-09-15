#!/bin/bash
# Baixa um dataset do Hugging Face para o HF_HOME compartilhado.
# Roda no LOGIN NODE: assume-se que os nos de compute NAO tem internet.
# No fim valida a leitura com HF_HUB_OFFLINE=1, lendo os parquet com pyarrow.
#
# Uso: scripts/prefetch_dataset.sh [REPO_ID] [CONFIG] [REVISION] [--no-validate]
#   Sem argumentos usa NT_BENCH_DATASET_ID / NT_BENCH_CONFIG /
#   NT_BENCH_DATASET_REVISION de environments/config.sh.
#   CONFIG e o SUBDIRETORIO do repositorio com os *.parquet (o repo do NT
#   benchmark nao declara configs do `datasets`; ex.: promoter_all).
#   Ao passar outro REPO_ID, passe tambem a REVISION (ex.: main).
#
# NAO usa o pacote `datasets`: o download e do huggingface_hub (ja no env) e a
# leitura e com pyarrow, instalado aqui se faltar, pela guarda de pip
# (environments/pip_guard.sh).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.sh"
source "$REPO_ROOT/environments/pip_guard.sh"

POS=()
VALIDATE=1
for arg in "$@"; do
    case "$arg" in
        --no-validate) VALIDATE=0 ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) POS+=("$arg") ;;
    esac
done
DATASET_ID="${POS[0]:-$NT_BENCH_DATASET_ID}"
CONFIG="${POS[1]:-$NT_BENCH_CONFIG}"
REVISION="${POS[2]:-$NT_BENCH_DATASET_REVISION}"

set +u
source "$CONDA_SH"
conda activate "$NT2_ENV_NAME"
set -u

export HF_HOME
mkdir -p "$HF_HOME"
echo "== Dataset:  $DATASET_ID"
echo "== Config:   $CONFIG (subdiretorio)"
echo "== Revision: $REVISION"
echo "== HF_HOME:  $HF_HOME"

if python -c "import pyarrow" 2>/dev/null; then
    echo "== pyarrow ja instalado: $(python -c 'import pyarrow; print(pyarrow.__version__)')"
else
    pip_install_protegido "prefetch_dataset: leitura de parquet" "$PYARROW_SPEC"
fi

INCLUDE=(--include "$CONFIG/*.parquet")
if command -v hf >/dev/null 2>&1; then
    hf download "$DATASET_ID" --repo-type dataset --revision "$REVISION" "${INCLUDE[@]}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "AVISO: 'hf' nao encontrado; usando o alias legado huggingface-cli."
    huggingface-cli download "$DATASET_ID" --repo-type dataset --revision "$REVISION" "${INCLUDE[@]}"
else
    echo "ERRO: nem 'hf' nem 'huggingface-cli' encontrados no env $NT2_ENV_NAME." >&2
    echo "      Rode environments/create_nt2_env.sh primeiro." >&2
    exit 1
fi

if [ "$VALIDATE" -eq 0 ]; then
    echo "== Download concluido (validacao offline pulada por --no-validate)."
    exit 0
fi

echo
echo "== Validando leitura OFFLINE (HF_HUB_OFFLINE=1, pyarrow) ..."
if HF_HUB_OFFLINE=1 python - "$DATASET_ID" "$CONFIG" "$REVISION" <<'PY'
import glob
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

dataset_id, config, revision = sys.argv[1:4]
snap = snapshot_download(dataset_id, repo_type="dataset", revision=revision,
                         allow_patterns=[f"{config}/*.parquet"], local_files_only=True)
arquivos = sorted(glob.glob(os.path.join(snap, config, "*.parquet")))
if not arquivos:
    sys.exit(f"nenhum {config}/*.parquet em {snap}")
for path in arquivos:
    t = pq.read_table(path)
    print(f"OK offline: {config}/{os.path.basename(path)}: {t.num_rows} linhas, colunas "
          f"{', '.join(f'{c.name}:{c.type}' for c in t.schema)}")
print(f"snapshot: {snap}")
PY
then
    echo "== Pronto: os parquet leem sem rede. Proximo passo (login node):"
    echo "   python $REPO_ROOT/scripts/benchmarks/nt_bench/prepare_data.py"
else
    echo "ERRO: o dataset NAO le offline. O preparo e o job vao falhar." >&2
    echo "      Confira se o HF_HOME acima e o mesmo que o job vai usar e se o" >&2
    echo "      download terminou sem erro." >&2
    exit 1
fi
