#!/bin/bash
# Guarda de `pip install` para o nt2-env. Este arquivo SO define funcoes; deve
# ser carregado com `source` depois do config.sh e com o env ja ativado:
#
#   source "$REPO_ROOT/environments/config.sh"
#   source "$REPO_ROOT/environments/pip_guard.sh"
#   pip_install_protegido "descricao curta" pacote "outro>=1.0" --index-url URL
#
# O env homologado e um ativo: mudancas nele tem que ser visiveis, nao
# implicitas. Conferir so a versao do transformers depois de um install nao
# basta; um downgrade silencioso de fsspec ou numpy quebra o torch e parece
# problema de modelo. Por isso toda instalacao no nt2-env passa por aqui:
#   1. grava `pip freeze` ANTES num arquivo temporario
#   2. roda `"$NT2_PYTHON" -m pip install "$@"`
#   3. grava `pip freeze` DEPOIS e compara
#   4. aborta (return 1), com diff explicito e comando de reversao, se algum
#      pacote de NT2_PROTECTED_PKGS que ja existia mudou de versao ou sumiu.
#      Ausente -> presente e permitido (e assim que o env e criado).
#   5. se algo mudou, anexa ao $WORK_DIR/env_versions.txt data, host,
#      interpretador, descricao, comando e a lista de mudancas (tambem quando
#      aborta).
#
# Interpretador: SEMPRE "$NT2_PYTHON" (config.sh), nunca `pip`/`python` do PATH.
# Com o PATH torto do cluster (/opt/conda/bin a frente do env), um `pip` puro
# instalaria no base e a guarda compararia o freeze do env errado sem perceber.

# Uso: pip_freeze_comparar ANTES DEPOIS
# Imprime uma linha por pacote que mudou (+ adicionado, ~ alterado, - removido)
# e, para protegidos que mudaram, o comando de reversao. Nada impresso = nada
# mudou. Codigo de saida: 0 ok, 3 pacote protegido que existia mudou ou sumiu.
pip_freeze_comparar() {
    "${NT2_PYTHON:?NT2_PYTHON nao definido; carregue environments/config.sh}" - \
        "$1" "$2" "${NT2_PROTECTED_PKGS:-}" "${TORCH_INDEX_URL:-}" <<'PY'
import re
import sys

antes_path, depois_path, protegidos_str, torch_index = sys.argv[1:5]


def norm(nome):
    return re.sub(r"[-_.]+", "-", nome).lower()


def ler(path):
    # Formatos do pip freeze: "nome==versao", "nome===versao", "nome @ url",
    # "-e <url>#egg=nome". Linhas de comentario/aviso sao ignoradas.
    pacotes = {}
    with open(path) as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            if linha.startswith("-e "):
                m = re.search(r"#egg=([A-Za-z0-9_.\-]+)", linha)
                nome, versao = (m.group(1) if m else linha), linha
            elif " @ " in linha:
                nome, versao = linha.split(" @ ", 1)
                versao = "@ " + versao
            elif "==" in linha:
                nome, versao = re.split(r"===?", linha, maxsplit=1)
            else:
                continue
            pacotes[norm(nome.strip())] = versao.strip()
    return pacotes


antes, depois = ler(antes_path), ler(depois_path)
protegidos = {norm(p) for p in protegidos_str.split()}
violacoes = []
for nome in sorted(set(antes) | set(depois)):
    va, vd = antes.get(nome), depois.get(nome)
    if va == vd:
        continue
    marca = " [PROTEGIDO]" if nome in protegidos else ""
    if va is None:
        print(f"  + {nome} {vd}{marca}")
    elif vd is None:
        print(f"  - {nome} {va} (removido){marca}")
    else:
        print(f"  ~ {nome} {va} -> {vd}{marca}")
    if nome in protegidos and va is not None:
        violacoes.append((nome, va))

if violacoes:
    print("  Para reverter os protegidos:")
    for nome, va in violacoes:
        # sys.executable == $NT2_PYTHON: este heredoc roda sob ele.
        if va.startswith("@ "):
            cmd = f'{sys.executable} -m pip install "{nome} {va}"'
        else:
            cmd = f'{sys.executable} -m pip install "{nome}=={va}"'
            if "+" in va and torch_index:  # build local (ex.: 2.11.0+cu128)
                cmd += f" --index-url {torch_index}"
        print(f"    {cmd}")
    sys.exit(3)
PY
}

# Anexa um registro de mudanca ao env_versions.txt.
# Uso: _pip_guard_registrar STATUS DESCRICAO MUDANCAS ARGS_DO_PIP...
_pip_guard_registrar() {
    local situacao="$1" descricao="$2" mudancas="$3"
    shift 3
    local arquivo="${WORK_DIR:?WORK_DIR nao definido; carregue environments/config.sh}/env_versions.txt"
    mkdir -p "$WORK_DIR"
    {
        echo
        echo "# ---- pip_install_protegido: $situacao em $(date -Iseconds) em $(hostname) ----"
        echo "env: ${CONDA_DEFAULT_ENV:-?}"
        echo "interpretador: $NT2_PYTHON"
        echo "descricao: $descricao"
        echo "comando: $NT2_PYTHON -m pip install $(printf '%q ' "$@")"
        echo "$mudancas"
    } >> "$arquivo"
    echo "== Mudanca registrada em $arquivo"
}

# Uso: pip_install_protegido DESCRICAO ARGS_DO_PIP...
pip_install_protegido() {
    local descricao="$1"
    shift
    if [ "${CONDA_DEFAULT_ENV:-}" != "${NT2_ENV_NAME:-}" ]; then
        echo "ERRO: pip_install_protegido chamado com o env '${CONDA_DEFAULT_ENV:-nenhum}' ativo," >&2
        echo "      esperado '${NT2_ENV_NAME:-?}'. Ative o env antes de instalar." >&2
        return 1
    fi
    if [ -z "${NT2_PYTHON:-}" ] || [ ! -x "$NT2_PYTHON" ]; then
        echo "ERRO: pip_install_protegido: interpretador '${NT2_PYTHON:-}' nao existe ou nao e" >&2
        echo "      executavel. Carregue environments/config.sh (ou exporte NT2_ENV_PREFIX)." >&2
        return 1
    fi
    echo "== Interpretador: $NT2_PYTHON"

    local antes depois mudancas
    local rc_pip=0 rc_cmp=0
    antes="$(mktemp)"
    depois="$(mktemp)"
    "$NT2_PYTHON" -m pip freeze > "$antes"
    echo "== $NT2_PYTHON -m pip install $* ($descricao)"
    "$NT2_PYTHON" -m pip install "$@" || rc_pip=$?
    "$NT2_PYTHON" -m pip freeze > "$depois"
    mudancas="$(pip_freeze_comparar "$antes" "$depois")" || rc_cmp=$?
    rm -f "$antes" "$depois"

    if [ "$rc_cmp" -ne 0 ] && [ "$rc_cmp" -ne 3 ]; then
        echo "ERRO: falha ao comparar os pip freeze (codigo $rc_cmp)." >&2
        return 1
    fi
    if [ -n "$mudancas" ]; then
        echo "== Mudancas no env $NT2_ENV_NAME:"
        echo "$mudancas"
    fi

    if [ "$rc_cmp" -eq 3 ]; then
        _pip_guard_registrar "ABORTADO (pacote protegido mudou)" "$descricao" "$mudancas" "$@"
        cat >&2 <<EOF

############################################################################
# ERRO: o pip install acima MUDOU pacote(s) protegido(s) do env homologado
# ($NT2_ENV_NAME). O env pode nao ser mais o que passou no smoke test.
# Reverta com os comandos listados acima e confira com:
#     $NT2_PYTHON -m pip freeze | grep -iE '$(echo "${NT2_PROTECTED_PKGS:-}" | tr ' ' '|')'
# Protegidos: ${NT2_PROTECTED_PKGS:-}
############################################################################

EOF
        return 1
    fi

    if [ "$rc_pip" -ne 0 ]; then
        [ -n "$mudancas" ] && _pip_guard_registrar "FALHOU (pip saiu com $rc_pip)" "$descricao" "$mudancas" "$@"
        echo "ERRO: pip install falhou com codigo $rc_pip ($descricao)." >&2
        return 1
    fi

    if [ -z "$mudancas" ]; then
        echo "== Nenhuma mudanca no env (requisitos ja satisfeitos)."
        return 0
    fi
    _pip_guard_registrar "OK" "$descricao" "$mudancas" "$@"
}
