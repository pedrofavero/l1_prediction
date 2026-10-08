# Dataset L1 (preparação no Mac)

Gera janelas de 1.024 bp, positivas (L1 e retrovírus) e negativas (TE, background, Markov), com
split sem vazamento, no contrato de CSV de `scripts/lib/nt2_train.py` (`sequence,label` primeiro).
Roda **no Mac**, não no cluster: o storage do cluster corrompeu arquivos grandes. Os CSVs prontos
são copiados para o cluster depois.

## Como rodar

```bash
# testes (FASTA/TSV sintéticos; o ponta a ponta usa o MMseqs2 e é pulado sem ele)
"$HOME/l1_data/venv/bin/python" -m unittest discover scripts/data_prep/l1/tests

# dry run (chr10 = dev, chr21 = test, chr22 = train; ~3 min), saída em $L1_DATASET_DIR/dryrun/
bash scripts/data_prep/l1/run_local.sh --limit-chroms chr10,chr21,chr22

# dataset completo: ident95 e depois ident98
bash scripts/data_prep/l1/run_local.sh
```

`run_local.sh` carrega `environments/config.local.sh` (perfil do Mac; o `config.sh` é o do
servidor), chama o interpretador por caminho absoluto (`"$PREP_PYTHON"`), repassa argumentos extras
ao `prepare_l1_dataset.py` e grava o log em `logs/prep/prep_l1_<timestamp>.log`. Com
`--only-if-changed`, um run é pulado quando parâmetros, sha256 das entradas, o próprio script e os
CSVs batem com o `data_meta.json` existente.

## Entradas (`config.local.sh`)

| Variável | Arquivo | Observação |
|---|---|---|
| `HG38_FA` | `genomes/hg38.fa` (+ `.fai`) | UCSC, soft-masked → `.upper()` |
| `RMSK_TXT` | `genomes/rmsk.txt.gz` | RepeatMasker UCSC, 0-based half-open |
| `L1FARM_TSV` | `l1farm/L1Farm_RGL1_BC_HG38.tsv` | sem cabeçalho, 0-based half-open (conferido: `End-Start == LocusLength`, o script aborta se não) |
| `RETRO_GENOMES_FA` | `retrovirus/retrovirus-refseq.fasta` | 92 genomas RefSeq |
| `RETRO_CDS_FA` | `retrovirus/retrovirus-refseq-cds.fasta` | 293 CDS (modo `pol`) |

Só chr1–22, chrX, chrY (contigs `_alt/_random/chrUn` e chrM ignorados).

## Saídas

`$L1_DATASET_DIR/ident95/` e `ident98/`: `train.csv`, `dev.csv`, `dev_strict.csv`, `test.csv`,
`test_strict.csv` e `data_meta.json`. Colunas (iguais em todos os arquivos):

```
sequence,label,source,detail,window_id,orientation,chrom,start,end,strand,group,gc,max_id_train,max_id_train_local
```

- `source` ∈ l1, retrovirus, te, background, markov. `detail`: `SUBFAMÍLIA:REGIÕES;…` (l1),
  accession (retrovirus), `repName/repClass` (te), vazio (background), `order5` (markov).
- Cada janela vira duas linhas: `orientation=fwd` (fita de referência como está) e `rc` (complemento
  reverso), sempre no mesmo arquivo. `strand` é a fita da anotação (`.` no background).
- `chrom/start/end` 0-based half-open (BED); vazios em markov e retrovirus.
- `gc` em fração (4 casas). Sequências em maiúsculas, só ACGT, exatamente 1.024 bp.
- `max_id_train` / `max_id_train_local`: identidade máxima com qualquer janela do train (vazias no
  train; `0.0000` sem hit). Ver "Identidade com o train".
- **`test_strict.csv` é a métrica principal**; `test.csv` é secundária. Para seleção de modelo, o
  análogo é `dev_strict.csv` (ver vazamento abaixo).

`data_meta.json`: parâmetros, sha256 das entradas e dos CSVs, commit do git, versões (Python,
pysam, numpy, MMseqs2), comandos exatos do MMseqs2 (clustering e busca), contagens por conjunto × source × label, GC médio
por source e por conjunto, composição dos `te` por repFamily, descartes (zona cinza, N, IUPAC),
déficits de amostragem e de balanço, relatório de vazamento (l1 fora do `*_strict` por
subfamília), faixas de `max_id_train*` e resultado das checagens.

## Definições

**Positivos (label 1)**
- `l1`: elemento = regiões L1Farm com `Identity >= --min-identity` no mesmo cromossomo e fita a
  ≤ 500 bp (componente conexo). Elemento < 1.024 bp → uma janela centrada; maior → passo `--stride`
  + uma janela final alinhada ao fim. Positiva se ≥ 200 bp cobertos pela união do L1Farm filtrado;
  1–199 bp = zona cinza (descartada). Janela com N → descartada.
- `retrovirus`: `--retrovirus-mode genome` (tiles 1.024/512 nos genomas) ou `pol` (só CDS cujo produto
  casa `pol`/`polymerase`/`reverse transcriptase`; a contagem vai para o log e o meta). Janela com
  caractere fora de ACGT → descartada.

**Negativos (label 0)** — 0 bp com qualquer L1 (rmsk, inclusive famílias `L1-*`, e L1Farm de
qualquer identidade), LTR (`repClass` LTR ou `LTR?`) e SVA (`repFamily` SVA ou `repClass`
Retroposon). Candidato que toca L1 conta como zona cinza, nunca vira negativo.
- `te` (35%): janela centrada em elementos rmsk de `--te-classes` (padrão `SINE,LINE,DNA`, igualdade
  exata: `DNA?`/`SINE?` ficam fora; LINE sem L1). LTR fora por padrão: ERVs têm o maquinário dos
  retrovírus usados como positivo.
- `background` (30%): tiles sem sobreposição com nenhuma repetição intercalada do rmsk (todas as
  classes exceto `Simple_repeat` e `Low_complexity`).
- `markov` (35%): cadeia de ordem `--markov-order` (pseudocontagem 1) treinada nas janelas `l1` dos
  cromossomos de train (fora do holdout e, no modo chrom, fora dos dev-chroms), nas duas orientações.
- `te`, `background` e `markov` são casados em GC (bins de 2 pontos) com o histograma do `l1` do
  mesmo papel de cromossomo (train, dev, test).

**Balanço**: negativos = positivos em cada conjunto, na proporção `--neg-mix`. Falta em `te` ou
`markov` é completada com `background`; qualquer déficit restante vai para o log e o meta. O balanço
de `dev_strict` e `test_strict` é recalculado dentro de cada um.

**Split sem vazamento**
1. `--holdout-chroms` (padrão chr8, chr21) → test. `--dev-mode chrom` (padrão): `--dev-chroms`
   (padrão chr7, chr10) → dev; precisam ser disjuntos do holdout (senão aborta).
2. MMseqs2 `easy-linclust --dbtype 2 --min-seq-id 0.90 -c 0.8 --cov-mode 0` sobre todas as janelas nas
   duas orientações (cópias em fitas opostas só se parecem como complemento reverso).
3. Union-find de cluster ∪ elemento L1 ∪ accession retroviral → `group`.
4. Janela genômica: o split vem do cromossomo (holdout → test, dev-chrom → dev, resto → train). Com
   `--dev-mode hash`, o dev sai dos grupos fora do holdout por hash determinístico (`--dev-frac`) e
   train/dev nunca dividem grupo. Retrovírus e Markov seguem o grupo: com janela genômica de train →
   train, senão de dev → dev; grupo só de retrovírus por `--retro-split` (0.8/0.1/0.1); só de Markov
   pelas frações estimadas dos positivos.
5. `dev_strict.csv` = janelas de dev cujo grupo não tem janela em train (no modo hash, == dev).
6. `--leak-policy`:
   - `filter-test` (padrão): train intacto; `test.csv` = tudo do holdout; `test_strict.csv` = só
     janelas de test cujo grupo não tem nenhuma janela em train/dev.
   - `purge-train`: janelas fora do holdout em grupos que tocam o test são removidas; `test_strict` ==
     `test`. O dev não é afetado.

Checagens que abortam antes de gravar: janela do train em nenhum outro arquivo; holdout só em test*,
dev-chroms só em dev*; `dev_strict` nunca divide `group` com train; `test_strict` nunca divide `group`
com train/dev; par fwd/rc nunca separado (a mesma janela só pode estar em {dev, dev_strict} ou
{test, test_strict}); sequência inválida; no modo hash, train e dev nunca dividem `group`. Em
`filter-test`, test e dev dividirem grupo com train é esperado e só reportado
(`checagens.grupo_em_test_e_train_dev`, `grupo_em_train_e_dev`).

**Identidade com o train**: uma execução de `easy-search --search-type 3 --strand 2 --min-seq-id 0.5
-c 0.19 --cov-mode 0` (0.19 ≈ 200 bp de 1.024, o mínimo de L1 de um positivo) de todas as janelas de
dev/test contra as do train. Do mesmo resultado:
- `max_id_train` = maior identidade entre hits com `qcov >= 0.8`, comparável ao critério do clustering
  e, portanto, ao `*_strict`;
- `max_id_train_local` = maior identidade entre alinhamentos ≥ 200 bp. Pega o caso de uma janela com
  só ~200 bp de L1 idêntico ao treino e flanco único.

O meta traz as faixas (`<0.80`, `0.80-0.90`, `0.90-0.95`, `>=0.95`) por conjunto × source para as
duas métricas, quantas janelas `l1` de test/dev têm `max_id_train < 0.90` (comparar com o `l1` do
`*_strict`) e quantas têm `max_id_train < 0.90` mas `max_id_train_local >= 0.95` (vazamento
escondido pela cobertura).

**Atenção — vazamento de L1 jovem**: L1HS/L1PA são quase idênticos entre cromossomos; a união
transitiva cria um grupo gigante. No run completo, o maior grupo reúne 81% do `l1` (ident95) e 79%
(ident98); cerca de 81% do `l1` de test e de dev cai fora do `*_strict`. Mesmo no `test_strict`,
quase todo `l1` tem `max_id_train_local >= 0.95` (854 de 1.037 no ident95, 300 de 301 no ident98):
os grupos separam janelas inteiras, mas trechos ≥ 200 bp quase idênticos ao train continuam lá.
Veja `vazamento.*_por_subfamilia` e `max_id_train` no meta.

## Determinismo e recursos

- Mesma seed → CSVs idênticos (sha256), conferido no dry run e no teste ponta a ponta. Toda etapa
  aleatória usa uma seed derivada de `--seed` + nome da etapa.
- O `easy-linclust` roda sempre com 1 thread: com várias threads, a pertença de algumas janelas aos
  clusters variou entre runs (mesmo número de grupos, membros diferentes). `--threads` vale só para o
  `easy-search`, que foi conferido como determinístico com 12 threads.
- Um cromossomo por vez em memória; temporários do MMseqs2 em `$PREP_TMP_DIR`, apagados ao fim.
