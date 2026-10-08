#!/bin/bash
# model_tag do baseline L1 -> familia, id, revision, origem dos pesos e --time do Slurm.
# Fonte unica para scripts/submit_l1_probe.sh (servidor) e scripts/train/l1/run_local_probe.sh
# (Mac): mesmas revisions nos dois. Ids e revisions vem do environments/config.sh, que
# precisa ter sido carregado antes. So define a funcao; nao executa nada.
#
#   source "$REPO_ROOT/scripts/l1_model_tags.sh"
#   resolver_model_tag nt2_50m   # define FAMILY MODEL_ID REVISION NOTA TEMPO; retorna 1 se desconhecido
#
# TEMPO e ESTIMATIVA (throughput a 1.024 bp ainda nao medido no servidor).

resolver_model_tag() {
    case "$1" in
        nt2_50m)
            FAMILY=nt2; MODEL_ID="$NT2_MODEL_ID"; REVISION="$NT2_MODEL_REVISION"; TEMPO=01:00:00
            NOTA="repositorio HF no revision pinado (model.safetensors do main)" ;;
        nt2_250m)
            FAMILY=nt2; MODEL_ID="$NT2_250M_ID"; REVISION="$NT2_250M_REVISION"; TEMPO=02:00:00
            NOTA="model.safetensors da conversao automatica do HF (SFconvertbot, refs/pr/3); demais arquivos identicos ao main c0f0359" ;;
        nt2_500m)
            FAMILY=nt2; MODEL_ID="$NT2_500M_ID"; REVISION="$NT2_500M_REVISION"; TEMPO=03:00:00
            NOTA="repositorio HF no revision pinado (model.safetensors do main)" ;;
        *)
            echo "ERRO: model_tag '$1' desconhecido (nt2_50m, nt2_250m, nt2_500m)" >&2
            return 1 ;;
    esac
}
