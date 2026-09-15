#!/bin/bash
# Cria ou completa o ambiente conda do Nucleotide Transformer v2 (nt2-env).
# Roda no LOGIN NODE (precisa de rede para o pip).
#
# Idempotente: se o env ja existir, apenas ativa e instala o que faltar.
# Nunca recria.
#
# Uso: environments/create_nt2_env.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.sh"
source "$REPO_ROOT/environments/pip_guard.sh"

echo "== Ambiente: $NT2_ENV_NAME (python $NT2_PYTHON_VERSION)"
echo "== REPO_ROOT=$REPO_ROOT"

# Os scripts de ativacao do conda podem referenciar variaveis nao definidas.
set +u
source "$CONDA_SH"
if conda env list | awk '{print $1}' | grep -qx "$NT2_ENV_NAME"; then
    echo "Env $NT2_ENV_NAME ja existe; apenas completando pacotes."
else
    echo "Criando env $NT2_ENV_NAME ..."
    conda create -y -n "$NT2_ENV_NAME" "python=$NT2_PYTHON_VERSION"
fi
conda activate "$NT2_ENV_NAME"
set -u

# O conda activate fica (define CONDA_PREFIX, que a guarda de pip confere), mas
# o PATH que ele produz nao e confiavel neste cluster: todo python/pip roda por
# "$NT2_PYTHON" (config.sh). Sem import check aqui: num env recem-criado o
# torch ainda nao existe.
if [[ ! -x "$NT2_PYTHON" ]]; then
    echo "ERRO: interpretador nao encontrado em $NT2_PYTHON" >&2
    echo "      (o conda ativou CONDA_PREFIX=${CONDA_PREFIX:-?})" >&2
    echo "Envs disponiveis:" >&2
    conda env list >&2 2>/dev/null || true
    echo "Se o env estiver em outro caminho, exporte NT2_ENV_PREFIX antes" >&2
    exit 1
fi
echo "== Python: $NT2_PYTHON"

# Todo pip install passa por environments/pip_guard.sh: freeze antes/depois,
# aborta se um pacote de NT2_PROTECTED_PKGS que ja existia mudar de versao e
# registra a mudanca em $WORK_DIR/env_versions.txt.

# PyTorch com CUDA 12.8 (wheel oficial). So instala se ainda nao importa.
if "$NT2_PYTHON" -c "import torch" 2>/dev/null; then
    echo "torch ja instalado: $("$NT2_PYTHON" -c 'import torch; print(torch.__version__)')"
else
    pip_install_protegido "create_nt2_env: torch cu128" torch --index-url "$TORCH_INDEX_URL"
fi

# Stack para inferencia e fine-tuning via Hugging Face.
# accelerate fica no env porque o fine-tuning real vai usar; o smoke_test.py
# NAO importa accelerate, datasets nem peft, para que uma incompatibilidade
# dessas libs nao vire um FAIL lido como falha de servidor.
#
# NAO instalar o pacote `nucleotide-transformer` da InstaDeep: ele arrasta
# JAX + haiku + scanpy e trava numpy<2 e pydantic==1.10.13. Para inferencia e
# fine-tuning via HF bastam torch, transformers, scikit-learn e numpy.
# Teto <5: a transformers 5 removeu find_pruneable_heads_and_indices de
# transformers.pytorch_utils, e o modeling_esm.py do NT importa essa funcao.
pip_install_protegido "create_nt2_env: stack HF" \
    "transformers>=4.52,<5" accelerate scikit-learn numpy huggingface_hub

# Diretorios que precisam existir ANTES do primeiro sbatch: o Slurm abre o
# arquivo de --output antes de executar a primeira linha do script, e logs/
# nao e versionado (num clone novo o job falharia sem deixar log).
mkdir -p "$REPO_ROOT/logs/smoke" "$WORK_DIR" "$HF_HOME"

# Espaco livre no filesystem do HF_HOME
echo
echo "== Espaco em HF_HOME=$HF_HOME"
df -h "$HF_HOME"
FREE_KB=$(df -Pk "$HF_HOME" | awk 'NR==2 {print $4}')
MIN_KB=$((10 * 1024 * 1024))
if [ "${FREE_KB:-0}" -lt "$MIN_KB" ]; then
    cat <<EOF

############################################################################
# AVISO: menos de 10 GB livres no filesystem de $HF_HOME
#
# O NT 50m (224 MB) cabe, mas o DNABERT-2 e principalmente o Evo 2 vao
# estourar uma home com cota apertada. Antes do prefetch, aponte o cache
# para a area de Storage do cluster:
#     export HF_HOME=/caminho/na/area/de/storage/huggingface
# e rode este script de novo para conferir.
############################################################################

EOF
fi

# Versoes efetivamente instaladas: na tela e ANEXADAS a $WORK_DIR/env_versions.txt
# (append: o arquivo guarda tambem o historico de mudancas da guarda de pip).
VERSIONS_FILE="$WORK_DIR/env_versions.txt"
echo "== Versoes instaladas"
{
    echo
    echo "# ==== snapshot do env gerado em $(date -Iseconds) em $(hostname) ===="
    echo "env: $NT2_ENV_NAME"
    "$NT2_PYTHON" - <<'PY'
import importlib
import sys

print(f"python {sys.version.split()[0]}")
print(f"sys.executable {sys.executable}")
for mod in ("torch", "transformers", "accelerate", "sklearn", "numpy", "huggingface_hub"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod} {m.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"{mod} NAO IMPORTA: {e}")
try:
    import torch
    print(f"torch.version.cuda {torch.version.cuda}")
    print(f"torch.cuda.is_available {torch.cuda.is_available()}  (no login node normalmente e False)")
except Exception:  # noqa: BLE001
    pass
PY
} | tee -a "$VERSIONS_FILE"
{
    echo
    echo "# ---- pip freeze ----"
    "$NT2_PYTHON" -m pip freeze
} >> "$VERSIONS_FILE"
echo
echo "Versoes anexadas a $VERSIONS_FILE (inclui pip freeze)."
echo "Proximo passo: scripts/prefetch_model.sh"
