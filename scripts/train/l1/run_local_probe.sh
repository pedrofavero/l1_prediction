#!/bin/bash
# Baseline L1 no Mac: verificacao -> extracao (MPS) -> probe, com o MESMO codigo e as
# MESMAS revisions do servidor. Roda de qualquer diretorio.
#
# Uso: scripts/train/l1/run_local_probe.sh <model_tag> <ident> [args do train_probe.py]
#   ex.: scripts/train/l1/run_local_probe.sh nt2_50m ident98
#
# Carrega config.local.sh (caminhos do Mac) e DEPOIS config.sh (ids/revisions): como o
# config.sh usa ${VAR:-default}, os caminhos do Mac vencem. Interpretador: "$ML_PYTHON"
# (environments/create_mac_ml_env.sh). Modelo tem que estar no HF_HOME (sem download
# implicito: HF_HUB_OFFLINE=1). Tudo sob `caffeinate -i` (o Mac nao dorme no meio).
# Log em logs/l1_probe/local_<tag>_<ident>_<timestamp>.log.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$AQUI" rev-parse --show-toplevel 2>/dev/null || (cd "$AQUI/../../.." && pwd))"
source "$REPO_ROOT/environments/config.local.sh"
source "$REPO_ROOT/environments/config.sh"
source "$REPO_ROOT/scripts/l1_model_tags.sh"

if [ $# -lt 2 ]; then
    sed -n '2,12p' "${BASH_SOURCE[0]}" >&2
    exit 1
fi
TAG="$1"
IDENT="$2"
shift 2
resolver_model_tag "$TAG" || exit 1

if [ ! -x "$ML_PYTHON" ]; then
    echo "ERRO: ML_PYTHON=$ML_PYTHON nao existe; rode environments/create_mac_ml_env.sh" >&2
    exit 1
fi
DATA_DIR="$L1_DATASET_DIR/$IDENT"
if [ ! -f "$DATA_DIR/data_meta.json" ]; then
    echo "ERRO: $DATA_DIR/data_meta.json nao existe" >&2
    exit 1
fi
EMB_DIR="$L1_EMB_DIR/$TAG/$IDENT"

export HF_HOME
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"

LOG_DIR="$REPO_ROOT/logs/l1_probe"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/local_${TAG}_${IDENT}_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"

{
    T0=$(date +%s)
    echo "== $(date -Iseconds) $TAG = $MODEL_ID@$REVISION ($FAMILY) | $IDENT | $ML_PYTHON"
    echo "== 1/3 integridade de $DATA_DIR"
    caffeinate -i "$ML_PYTHON" "$AQUI/extract_embeddings.py" --verify-only --data-dir "$DATA_DIR"
    echo "== 2/3 extracao -> $EMB_DIR"
    caffeinate -i "$ML_PYTHON" "$AQUI/extract_embeddings.py" --model-tag "$TAG" --family "$FAMILY" \
        --model-id "$MODEL_ID" --revision "$REVISION" --ident "$IDENT" --device auto \
        --data-dir "$DATA_DIR" --out-dir "$EMB_DIR" --weights-note "$NOTA"
    T1=$(date +%s)
    echo "== 3/3 probe -> $L1_PROBE_DIR/$TAG/$IDENT"
    caffeinate -i "$ML_PYTHON" "$AQUI/train_probe.py" --model-tag "$TAG" --ident "$IDENT" \
        --data-dir "$DATA_DIR" --emb-dir "$EMB_DIR" --out-root "$L1_PROBE_DIR" "$@"
    T2=$(date +%s)
    echo "== tempo: extracao (com verificacao) $((T1 - T0)) s, probe $((T2 - T1)) s, total $((T2 - T0)) s"
} 2>&1 | tee "$LOG"
