#!/bin/bash
# Submete o benchmark NT (etapa 2 da homologacao) na particao shared. Roda no
# LOGIN NODE, de qualquer diretorio.
#
# Uso: scripts/submit_bench.sh [args do run_benchmark.py]
#   ex.: scripts/submit_bench.sh --lr 5e-5 --epochs 3
#
# Cria logs/bench antes de submeter (o Slurm abre o arquivo de --output antes
# de executar o script) e passa caminhos ABSOLUTOS de --output/--error, que
# sobrescrevem as diretivas relativas do .sbatch.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.sh"

LOG_DIR="$REPO_ROOT/logs/bench"
mkdir -p "$LOG_DIR"

JOB_ID=$(sbatch --parsable \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    "$REPO_ROOT/slurm/job_bench_nt2.sbatch" "$@")
JOB_ID="${JOB_ID%%;*}"   # --parsable pode devolver "id;cluster"

echo "Job submetido: $JOB_ID"
squeue -j "$JOB_ID" || true
echo
echo "Acompanhar:"
echo "  tail -f $LOG_DIR/bench_nt2_${JOB_ID}.out"
echo "  tail -f $LOG_DIR/bench_nt2_${JOB_ID}.err"
echo "Relatorio ao final: $NT_BENCH_REPORT_DIR/run_${JOB_ID}/benchmark_report.json"
echo "Predicoes do test:  $NT_BENCH_REPORT_DIR/run_${JOB_ID}/predicoes_test.csv"
