#!/bin/bash
# Baixa um modelo do Hugging Face para o HF_HOME compartilhado.
# Roda no LOGIN NODE: assume-se que os nos de compute NAO tem internet.
# No fim valida a carga com HF_HUB_OFFLINE=1 para provar que o job vai
# conseguir carregar sem rede.
#
# Uso: scripts/prefetch_model.sh [MODEL_ID] [REVISION] [--no-validate]
#   Sem argumentos usa NT2_MODEL_ID / NT2_MODEL_REVISION de environments/config.sh.
#   --no-validate pula a validacao offline (util para modelos que nao carregam
#   com AutoModelForMaskedLM; ex.: DNABERT-2 precisa de BertConfig explicito).
#
# O mesmo script serve para o NT-250m/500m e para o DNABERT-2:
#   scripts/prefetch_model.sh InstaDeepAI/nucleotide-transformer-v2-500m-multi-species main
#   scripts/prefetch_model.sh zhihan1996/DNABERT-2-117M main --no-validate
# O --exclude abaixo e inofensivo para repositorios que nao tem esses arquivos.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd))"
source "$REPO_ROOT/environments/config.sh"

POS=()
VALIDATE=1
for arg in "$@"; do
    case "$arg" in
        --no-validate) VALIDATE=0 ;;
        -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) POS+=("$arg") ;;
    esac
done
MODEL_ID="${POS[0]:-$NT2_MODEL_ID}"
REVISION="${POS[1]:-$NT2_MODEL_REVISION}"

set +u
source "$CONDA_SH"
conda activate "$NT2_ENV_NAME"
set -u

export HF_HOME
mkdir -p "$HF_HOME"
echo "== Modelo:   $MODEL_ID"
echo "== Revision: $REVISION"
echo "== HF_HOME:  $HF_HOME"

# Duplicatas que nao servem ao PyTorch: pesos TF (*.h5) e JAX (jax_model/*).
EXCLUDE=(--exclude "*.h5" "jax_model/*")
if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_ID" --revision "$REVISION" "${EXCLUDE[@]}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "AVISO: 'hf' nao encontrado; usando o alias legado huggingface-cli."
    huggingface-cli download "$MODEL_ID" --revision "$REVISION" "${EXCLUDE[@]}"
else
    echo "ERRO: nem 'hf' nem 'huggingface-cli' encontrados no env $NT2_ENV_NAME." >&2
    echo "      Rode environments/create_nt2_env.sh primeiro." >&2
    exit 1
fi

if [ "$VALIDATE" -eq 0 ]; then
    echo "== Download concluido (validacao offline pulada por --no-validate)."
    exit 0
fi

echo
echo "== Validando carga OFFLINE (HF_HUB_OFFLINE=1, CPU) ..."
if HF_HUB_OFFLINE=1 python - "$MODEL_ID" "$REVISION" <<'PY'
import sys
import time

model_id, revision = sys.argv[1:3]
from transformers import AutoModelForMaskedLM, AutoTokenizer

t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)
n = sum(p.numel() for p in model.parameters())
print(f"OK offline: {model_id}@{revision[:12]} vocab={len(tok)} params={n / 1e6:.1f}M em {time.time() - t0:.1f}s")
PY
then
    echo "== Pronto: o job vai conseguir carregar o modelo sem rede."
else
    echo "ERRO: o modelo NAO carrega offline. O job na particao shared vai falhar." >&2
    echo "      Confira se o HF_HOME acima e o mesmo que o job vai usar e se o" >&2
    echo "      download terminou sem erro." >&2
    exit 1
fi
