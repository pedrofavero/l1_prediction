#!/bin/bash -l
#SBATCH --job-name=bench_nt2
#SBATCH --partition=shared
#SBATCH --gres=mps:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench/%x_%j.out
#SBATCH --error=logs/bench/%x_%j.err
#
# Etapa 2 da homologacao: fine-tuning do NT v2 50m no benchmark NT promoter_all
# (particao shared, ~1/4 dos SMs via MPS; a VRAM NAO e particionada).
# O smoke test mediu 1,18 GB de VRAM e 163 seq/s a batch 16 com mps:14; ~48k
# seqs x 4 epocas cabem com folga em 2 h.
#
# NUNCA trocar --gres=mps: por --gres=gpu: nesta particao, nem combinar os dois.
#
# Os caminhos de --output/--error acima sao relativos ao diretorio de submissao
# e servem de fallback para `sbatch slurm/job_bench_nt2.sbatch` chamado da raiz
# do repositorio. O caminho normal e scripts/submit_bench.sh, que cria
# logs/bench e passa caminhos absolutos.
#
# Uso: scripts/submit_bench.sh [args do run_benchmark.py, ex.: --lr 5e-5 --epochs 3]
# Exit code: o do run_benchmark.py (0 PASS, 1 FAIL, 2 WARN), ou 1 se o preparo
# dos dados falhar.
set -euo pipefail

# No sbatch, BASH_SOURCE aponta para a copia no spool do Slurm. O wrapper
# exporta REPO_ROOT; sem ele, tenta o git a partir do cwd (= diretorio de
# submissao) e por fim o proprio diretorio de submissao.
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$REPO_ROOT/environments/config.sh"

echo "== Job ${SLURM_JOB_ID:-?} em $(hostname), $(date -Iseconds)"
echo "== REPO_ROOT=$REPO_ROOT"
echo "== HF_HOME=$HF_HOME"

set +u
source "$CONDA_SH"
conda activate "$NT2_ENV_NAME"
set -u

export HF_HOME
export HF_HUB_OFFLINE=1          # o no de compute nao tem rede; tudo vem do cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"   # lib/nt2_train.py
export NT2_MODEL_ID NT2_MODEL_REVISION
export NT_BENCH_DATASET_ID NT_BENCH_CONFIG NT_BENCH_DATASET_REVISION NT_BENCH_DATA_DIR NT_BENCH_REPORT_DIR

nvidia-smi || echo "AVISO: nvidia-smi indisponivel neste no"

BENCH_DIR="$REPO_ROOT/scripts/benchmarks/nt_bench"
# Gera os CSV se nao existirem ou se nao baterem com a configuracao (parametros
# e sha256 do data_meta.json). Com dados atuais, nao importa nada alem da stdlib.
echo "== Dados do benchmark em $NT_BENCH_DATA_DIR"
if ! python "$BENCH_DIR/prepare_data.py" --only-if-changed; then
    echo "ERRO no PREPARO DOS DADOS (nao e problema de modelo). Rode no login node:" >&2
    echo "    scripts/prefetch_dataset.sh" >&2
    echo "    python scripts/benchmarks/nt_bench/prepare_data.py" >&2
    exit 1
fi

# Sob `set -e`, um `python ...; rc=$?` nunca chegaria ao rc em caso de falha.
rc=0
python "$BENCH_DIR/run_benchmark.py" "$@" || rc=$?
echo "== run_benchmark.py terminou com codigo $rc ($(date -Iseconds))"
exit $rc
