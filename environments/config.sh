#!/bin/bash
# Configuracao central do repositorio l1-prediction.
#
# Este arquivo SO define variaveis; nao executa nada. Deve ser carregado com
# `source` por qualquer script .sh do repositorio, depois que o chamador
# resolveu REPO_ROOT:
#
#   REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
#   source "$REPO_ROOT/environments/config.sh"
#
# Toda variavel pode ser sobrescrita exportando-a ANTES do source
# (ex.: `export HF_HOME=/outro/caminho` e depois `scripts/prefetch_model.sh`).

# Raiz do repositorio (fallback: a pasta acima de environments/)
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Conda do cluster CISIA
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"

# Ambiente conda do Nucleotide Transformer v2
NT2_ENV_NAME="${NT2_ENV_NAME:-nt2-env}"
NT2_PYTHON_VERSION="${NT2_PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Cache do Hugging Face. Os modelos sao baixados no login node (com rede) e
# lidos offline pelos jobs (HF_HUB_OFFLINE=1), entao o mesmo HF_HOME precisa
# estar visivel nos nos de compute.
# PENDENCIA: a area de storage do cluster (Home, Projects, Datasets ou
# Storage-<usuario>) ainda nao foi definida. Se a home tiver cota apertada,
# exportar HF_HOME apontando para a area de Storage antes do prefetch.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Modelo do smoke test: o menor da familia NT v2. O teste e do servidor, nao do
# modelo. Revision verificada em 2026-09-03.
NT2_MODEL_ID="${NT2_MODEL_ID:-InstaDeepAI/nucleotide-transformer-v2-50m-multi-species}"
NT2_MODEL_REVISION="${NT2_MODEL_REVISION:-81b29e5786726d891dbf929404ef20adca5b36f1}"

# Diretorio de trabalho FORA do repositorio (datasets gerados, relatorios,
# versoes do env). Nada disso e versionado.
# PENDENCIA: mesma decisao de storage acima.
WORK_DIR="${WORK_DIR:-$HOME/l1_prediction_work}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-$WORK_DIR/toy_dataset}"
SMOKE_REPORT_DIR="${SMOKE_REPORT_DIR:-$WORK_DIR/smoke_nt2}"

export REPO_ROOT CONDA_SH NT2_ENV_NAME NT2_PYTHON_VERSION TORCH_INDEX_URL
export HF_HOME NT2_MODEL_ID NT2_MODEL_REVISION
export WORK_DIR SMOKE_DATA_DIR SMOKE_REPORT_DIR
