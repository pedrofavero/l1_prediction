#!/bin/bash -l
#SBATCH --job-name=smoke_nt2
#SBATCH --partition=shared
#SBATCH --gres=mps:14
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke/%x_%j.out
#SBATCH --error=logs/smoke/%x_%j.err
#
# Smoke test de infraestrutura: NT v2 50m na particao shared (fatia de GPU
# via MPS, ~1/7 dos SMs; a VRAM NAO e particionada).
#
# NUNCA trocar --gres=mps: por --gres=gpu: nesta particao, nem combinar os dois.
#
# Os caminhos de --output/--error acima sao relativos ao diretorio de submissao
# e servem de fallback para `sbatch slurm/job_smoke_nt2.sbatch` chamado da raiz
# do repositorio. O caminho normal e scripts/submit_smoke.sh, que cria
# logs/smoke e passa caminhos absolutos.
#
# Uso: scripts/submit_smoke.sh [--quick]
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
export NT2_MODEL_ID NT2_MODEL_REVISION SMOKE_DATA_DIR SMOKE_REPORT_DIR

nvidia-smi || echo "AVISO: nvidia-smi indisponivel neste no"

SMOKE_DIR="$REPO_ROOT/scripts/smoke_tests/nt2_infra"
# Regera o dataset se nao existir ou se dataset_meta.json nao bater com a
# configuracao atual do gerador (motivos, seed, seq_len, tamanhos). Um dataset
# antigo nao sobrevive silenciosamente a uma mudanca no gerador.
echo "== Dataset sintetico em $SMOKE_DATA_DIR"
python "$SMOKE_DIR/make_toy_dataset.py" --out-dir "$SMOKE_DATA_DIR" --only-if-changed

# Sob `set -e`, um `python ...; rc=$?` nunca chegaria ao rc em caso de falha.
rc=0
python "$SMOKE_DIR/smoke_test.py" "$@" || rc=$?
echo "== smoke_test.py terminou com codigo $rc ($(date -Iseconds))"
exit $rc
