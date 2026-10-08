#!/bin/bash
# Perfil do MAC para a preparacao de dados (scripts/data_prep/l1). O config.sh
# e o perfil do servidor e nao e carregado aqui.
#
# Este arquivo SO define variaveis; nao executa nada. Carregar com `source`
# depois de resolver REPO_ROOT:
#
#   source "$REPO_ROOT/environments/config.local.sh"
#
# Toda variavel pode ser sobrescrita exportando-a ANTES do source.
# A preparacao roda no Mac porque o storage do cluster corrompeu arquivos grandes.

: "${HOME:?HOME nao definido — impossivel derivar WORK_DIR}"

# Diretorio de trabalho FORA do repositorio (entradas, venv, saidas). Nada disso
# e versionado.
WORK_DIR="${WORK_DIR:-$HOME/l1_data}"

# Interpretador e binarios por caminho ABSOLUTO, nunca pelo PATH.
# venv: Python 3.13, pysam 0.24.1, numpy 2.5.3. MMseqs2: binario osx-universal.
PREP_PYTHON="${PREP_PYTHON:-$WORK_DIR/venv/bin/python}"
MMSEQS_BIN="${MMSEQS_BIN:-$WORK_DIR/bin/mmseqs/bin/mmseqs}"

# Entradas
HG38_FA="${HG38_FA:-$WORK_DIR/genomes/hg38.fa}"                 # UCSC, soft-masked, com .fai
RMSK_TXT="${RMSK_TXT:-$WORK_DIR/genomes/rmsk.txt.gz}"           # RepeatMasker UCSC, 0-based
L1FARM_TSV="${L1FARM_TSV:-$WORK_DIR/l1farm/L1Farm_RGL1_BC_HG38.tsv}"  # 0-based half-open
RETRO_GENOMES_FA="${RETRO_GENOMES_FA:-$WORK_DIR/retrovirus/retrovirus-refseq.fasta}"
RETRO_CDS_FA="${RETRO_CDS_FA:-$WORK_DIR/retrovirus/retrovirus-refseq-cds.fasta}"

# Saidas (ident95/, ident98/) e temporarios do MMseqs2 (apagados ao fim de cada run)
L1_DATASET_DIR="${L1_DATASET_DIR:-$WORK_DIR/l1_dataset}"
PREP_TMP_DIR="${PREP_TMP_DIR:-$WORK_DIR/tmp}"

export WORK_DIR PREP_PYTHON MMSEQS_BIN
export HG38_FA RMSK_TXT L1FARM_TSV RETRO_GENOMES_FA RETRO_CDS_FA
export L1_DATASET_DIR PREP_TMP_DIR
