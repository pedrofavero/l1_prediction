#!/bin/bash
# Gera o dataset L1 no Mac: ident95 e depois ident98. Roda de qualquer diretorio.
#
# Uso: scripts/data_prep/l1/run_local.sh [args extras do prepare_l1_dataset.py]
#   ex.: scripts/data_prep/l1/run_local.sh --limit-chroms chr21,chr22   (dry run)
#
# Saida em $L1_DATASET_DIR/ident95 e ident98. Com --limit-chroms a base vira
# $L1_DATASET_DIR/dryrun, para o dry run nunca sobrescrever o dataset real.
# Log completo em logs/prep/prep_l1_<timestamp>.log.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$AQUI" rev-parse --show-toplevel 2>/dev/null || (cd "$AQUI/../../.." && pwd))"
source "$REPO_ROOT/environments/config.local.sh"

if [ ! -x "$PREP_PYTHON" ]; then
    echo "ERRO: PREP_PYTHON=$PREP_PYTHON nao existe ou nao e executavel" >&2
    exit 1
fi

OUT_BASE="$L1_DATASET_DIR"
for a in "$@"; do
    case "$a" in
        --limit-chroms|--limit-chroms=*) OUT_BASE="$L1_DATASET_DIR/dryrun" ;;
    esac
done

LOG_DIR="$REPO_ROOT/logs/prep"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/prep_l1_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"

{
    for ident in 95 98; do
        echo "=== ident$ident ==="
        "$PREP_PYTHON" "$AQUI/prepare_l1_dataset.py" \
            --min-identity "0.$ident" --out-dir "$OUT_BASE/ident$ident" "$@"
    done
} 2>&1 | tee "$LOG"
