#!/bin/bash
# Submete o baseline L1 (embeddings congelados + probe) na particao shared. Roda
# no LOGIN NODE, de qualquer diretorio.
#
# Uso: scripts/submit_l1_probe.sh <model_tag> <ident> [args do train_probe.py]
#   model_tag: nt2_50m | nt2_250m | nt2_500m   (id e revision pinados no config.sh)
#   ident:     ident95 | ident98
#   ex.: scripts/submit_l1_probe.sh nt2_50m ident98
#        scripts/submit_l1_probe.sh nt2_500m ident95 --mlp
#
# --time por modelo e ESTIMATIVA (throughput a 1.024 bp ainda nao medido); revisar
# com o seq_por_s gravado no meta.json dos embeddings depois do primeiro run.
# O modelo tem que estar no cache (HF_HOME) antes: scripts/prefetch_model.sh.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.sh"

if [ $# -lt 2 ]; then
    sed -n '2,13p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi
TAG="$1"
IDENT="$2"
shift 2

source "$REPO_ROOT/scripts/l1_model_tags.sh"
resolver_model_tag "$TAG" || exit 1
case "$IDENT" in
    ident95|ident98) ;;
    *) echo "ERRO: ident '$IDENT' desconhecido (ident95, ident98)" >&2; exit 1 ;;
esac
if [ ! -f "$L1_DATASET_DIR/$IDENT/data_meta.json" ]; then
    echo "ERRO: $L1_DATASET_DIR/$IDENT/data_meta.json nao existe" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/logs/l1_probe"
mkdir -p "$LOG_DIR"
NOME="l1probe_${TAG}_${IDENT}"

export REPO_ROOT
export L1_MODEL_TAG="$TAG" L1_MODEL_FAMILY="$FAMILY" L1_MODEL_ID="$MODEL_ID" L1_MODEL_REVISION="$REVISION"
export L1_MODEL_WEIGHTS_NOTE="$NOTA" L1_IDENT="$IDENT"

JOB_ID=$(sbatch --parsable \
    --job-name="$NOME" \
    --time="$TEMPO" \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    "$REPO_ROOT/slurm/job_l1_probe.sbatch" "$@")
JOB_ID="${JOB_ID%%;*}"   # --parsable pode devolver "id;cluster"

echo "Job submetido: $JOB_ID ($TAG = $MODEL_ID@${REVISION:0:12}, $IDENT, --time $TEMPO)"
squeue -j "$JOB_ID" || true
echo
echo "Acompanhar:"
echo "  tail -f $LOG_DIR/${NOME}_${JOB_ID}.out"
echo "  tail -f $LOG_DIR/${NOME}_${JOB_ID}.err"
echo "Embeddings: $L1_EMB_DIR/$TAG/$IDENT/"
echo "Relatorio:  $L1_PROBE_DIR/$TAG/$IDENT/<timestamp>/report.json"
