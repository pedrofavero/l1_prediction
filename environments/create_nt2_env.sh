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

# PyTorch com CUDA 12.8 (wheel oficial). So instala se ainda nao importa.
if python -c "import torch" 2>/dev/null; then
    echo "torch ja instalado: $(python -c 'import torch; print(torch.__version__)')"
else
    pip install torch --index-url "$TORCH_INDEX_URL"
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
pip install "transformers>=4.52,<5" accelerate scikit-learn numpy huggingface_hub

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

# Versoes efetivamente instaladas: na tela e em $WORK_DIR/env_versions.txt
VERSIONS_FILE="$WORK_DIR/env_versions.txt"
echo "== Versoes instaladas"
{
    echo "# gerado em $(date -Iseconds) em $(hostname)"
    echo "env: $NT2_ENV_NAME"
    python - <<'PY'
import importlib
import sys

print(f"python {sys.version.split()[0]}")
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
} | tee "$VERSIONS_FILE"
{
    echo
    echo "# ---- pip freeze ----"
    pip freeze
} >> "$VERSIONS_FILE"
echo
echo "Versoes gravadas em $VERSIONS_FILE (inclui pip freeze)."
echo "Proximo passo: scripts/prefetch_model.sh"
