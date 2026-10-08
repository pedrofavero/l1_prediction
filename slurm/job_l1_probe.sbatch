#!/bin/bash -l
#SBATCH --job-name=l1probe
#SBATCH --partition=shared
#SBATCH --gres=mps:14
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/l1_probe/%x_%j.out
#SBATCH --error=logs/l1_probe/%x_%j.err
#
# Baseline L1 com embeddings congelados, num job so:
#   1. integridade dos CSVs (sha256 vs data_meta.json) -> aborta antes da GPU
#   2. extracao dos embeddings (GPU, ~1/7 dos SMs via MPS; a VRAM NAO e particionada)
#   3. probe (regressao logistica em CPU) + avaliacao + resumo
# Splits ja extraidos com o mesmo modelo/revision/CSV sao reaproveitados.
#
# NUNCA trocar --gres=mps: por --gres=gpu: nesta particao, nem combinar os dois.
# --time e o --job-name vem do scripts/submit_l1_probe.sh (por model_tag); o
# valor acima e fallback. Sem --mail-user: o Slurm usa a conta de quem submete.
#
# Uso: scripts/submit_l1_probe.sh <model_tag> <ident> [args do train_probe.py, ex.: --mlp]
# O wrapper exporta L1_MODEL_TAG, L1_MODEL_FAMILY, L1_MODEL_ID, L1_MODEL_REVISION,
# L1_MODEL_WEIGHTS_NOTE e L1_IDENT.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$REPO_ROOT/environments/config.sh"

: "${L1_MODEL_TAG:?use scripts/submit_l1_probe.sh}" "${L1_MODEL_FAMILY:?}" "${L1_MODEL_ID:?}"
: "${L1_MODEL_REVISION:?}" "${L1_IDENT:?}"

echo "== Job ${SLURM_JOB_ID:-?} em $(hostname), $(date -Iseconds)"
echo "== REPO_ROOT=$REPO_ROOT"
echo "== Modelo: $L1_MODEL_TAG = $L1_MODEL_ID@$L1_MODEL_REVISION ($L1_MODEL_FAMILY) | dataset $L1_IDENT"
echo "== HF_HOME=$HF_HOME"

set +u
source "$CONDA_SH"
conda activate "$NT2_ENV_NAME"
set -u

# O PATH do conda activate nao e confiavel neste cluster: todo python roda por "$NT2_PYTHON".
if [[ ! -x "$NT2_PYTHON" ]]; then
    echo "ERRO: interpretador nao encontrado em $NT2_PYTHON" >&2
    conda env list >&2 2>/dev/null || true
    exit 1
fi
echo "== Python: $NT2_PYTHON"
"$NT2_PYTHON" -c "import numpy, sklearn, torch, transformers" || {
    echo "ERRO: nt2-env incompleto ou interpretador errado" >&2
    exit 1
}

export HF_HOME
export HF_HUB_OFFLINE=1          # o no de compute nao tem rede; tudo vem do cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"   # lib/l1_dados.py

DATA_DIR="$L1_DATASET_DIR/$L1_IDENT"
EMB_DIR="$L1_EMB_DIR/$L1_MODEL_TAG/$L1_IDENT"
TRAIN_DIR="$REPO_ROOT/scripts/train/l1"

echo "== 1/3 integridade de $DATA_DIR"
"$NT2_PYTHON" "$TRAIN_DIR/extract_embeddings.py" --verify-only --data-dir "$DATA_DIR"

nvidia-smi || echo "AVISO: nvidia-smi indisponivel neste no"

echo "== 2/3 extracao -> $EMB_DIR"
"$NT2_PYTHON" "$TRAIN_DIR/extract_embeddings.py" --model-tag "$L1_MODEL_TAG" --family "$L1_MODEL_FAMILY" \
    --model-id "$L1_MODEL_ID" --revision "$L1_MODEL_REVISION" --ident "$L1_IDENT" \
    --data-dir "$DATA_DIR" --out-dir "$EMB_DIR" --weights-note "${L1_MODEL_WEIGHTS_NOTE:-}"

echo "== 3/3 probe -> $L1_PROBE_DIR/$L1_MODEL_TAG/$L1_IDENT"
# Sob `set -e`, um `"$NT2_PYTHON" ...; rc=$?` nunca chegaria ao rc em caso de falha.
rc=0
"$NT2_PYTHON" "$TRAIN_DIR/train_probe.py" --model-tag "$L1_MODEL_TAG" --ident "$L1_IDENT" \
    --data-dir "$DATA_DIR" --emb-dir "$EMB_DIR" --out-root "$L1_PROBE_DIR" "$@" || rc=$?
echo "== train_probe.py terminou com codigo $rc ($(date -Iseconds))"
exit $rc
