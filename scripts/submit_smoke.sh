#!/bin/bash
# Submete o smoke test do NT v2 na particao shared. Roda no LOGIN NODE, de
# qualquer diretorio.
#
# Uso: scripts/submit_smoke.sh [--quick]
#   --quick e repassado ao smoke_test.py (pula o fine-tuning).
#
# Cria logs/smoke antes de submeter (o Slurm abre o arquivo de --output antes
# de executar o script) e passa caminhos ABSOLUTOS de --output/--error, que
# sobrescrevem as diretivas relativas do .sbatch.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.sh"

LOG_DIR="$REPO_ROOT/logs/smoke"
mkdir -p "$LOG_DIR"

JOB_ID=$(sbatch --parsable \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    "$REPO_ROOT/slurm/job_smoke_nt2.sbatch" "$@")
JOB_ID="${JOB_ID%%;*}"   # --parsable pode devolver "id;cluster"

echo "Job submetido: $JOB_ID"
squeue -j "$JOB_ID" || true
echo
echo "Acompanhar:"
echo "  tail -f $LOG_DIR/smoke_nt2_${JOB_ID}.out"
echo "  tail -f $LOG_DIR/smoke_nt2_${JOB_ID}.err"
echo "Relatorio ao final: $SMOKE_REPORT_DIR/smoke_report.json"
