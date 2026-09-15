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

# Interpretador do env por caminho ABSOLUTO. Nunca confiar no `python`/`pip` do
# PATH: no no de compute do CISIA o PATH pode ficar com /opt/conda/bin a frente
# do env mesmo depois de `conda activate`, que retorna 0 assim mesmo. `python`
# roda entao o base (sem numpy/torch) e a falha e silenciosa ate o primeiro
# import (job 2108). Todo script chama "$NT2_PYTHON" e "$NT2_PYTHON" -m pip.
: "${HOME:?HOME nao definido — impossivel derivar NT2_ENV_PREFIX}"
NT2_ENV_PREFIX="${NT2_ENV_PREFIX:-$HOME/.conda/envs/$NT2_ENV_NAME}"
NT2_PYTHON="${NT2_PYTHON:-$NT2_ENV_PREFIX/bin/python}"

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

# Benchmark da etapa 2 (validacao do pipeline contra numero publicado).
# O repositorio do HF NAO declara configs (card YAML vazio; o datasets-server so
# ve "default", que mistura as 18 tarefas): "config" aqui e o SUBDIRETORIO com
# train.parquet/test.parquet. Revision verificada em 2026-09-15.
NT_BENCH_DATASET_ID="${NT_BENCH_DATASET_ID:-InstaDeepAI/nucleotide_transformer_downstream_tasks}"
NT_BENCH_CONFIG="${NT_BENCH_CONFIG:-promoter_all}"
NT_BENCH_DATASET_REVISION="${NT_BENCH_DATASET_REVISION:-96d86d567d4cd33536e49b429dc7983121619a08}"
NT_BENCH_DATA_DIR="${NT_BENCH_DATA_DIR:-$WORK_DIR/nt_bench/$NT_BENCH_CONFIG/data}"
NT_BENCH_REPORT_DIR="${NT_BENCH_REPORT_DIR:-$WORK_DIR/nt_bench/$NT_BENCH_CONFIG/runs}"

# Leitura dos parquet do benchmark. NAO usar o pacote `datasets` no nt2-env: ele
# arrasta fsspec/pandas/dill/multiprocess e o fsspec interage com huggingface_hub
# e torch. `hf download` + pyarrow bastam. Se o `datasets` for mesmo necessario
# um dia, vai para um env separado.
PYARROW_SPEC="${PYARROW_SPEC:-pyarrow>=14}"

# Stack homologado: environments/pip_guard.sh aborta qualquer pip install que
# mude a versao de um destes (nomes normalizados: minusculas, `-`).
NT2_PROTECTED_PKGS="${NT2_PROTECTED_PKGS:-torch transformers tokenizers huggingface-hub numpy scikit-learn accelerate}"

export REPO_ROOT CONDA_SH NT2_ENV_NAME NT2_PYTHON_VERSION TORCH_INDEX_URL
export NT2_ENV_PREFIX NT2_PYTHON
export HF_HOME NT2_MODEL_ID NT2_MODEL_REVISION
export WORK_DIR SMOKE_DATA_DIR SMOKE_REPORT_DIR
export NT_BENCH_DATASET_ID NT_BENCH_CONFIG NT_BENCH_DATASET_REVISION
export NT_BENCH_DATA_DIR NT_BENCH_REPORT_DIR PYARROW_SPEC NT2_PROTECTED_PKGS
