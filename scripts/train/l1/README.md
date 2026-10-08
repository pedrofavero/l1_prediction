# Baseline L1: embeddings congelados + probe

Mesmo classificador (regressão logística, C escolhido no dev) para todos os modelos, sobre embeddings
congelados. Fine-tuning é outra etapa.

## Como rodar

```bash
# no Mac: testes (adaptador falso, sem GPU nem torch; precisa de scikit-learn no ~/l1_data/venv)
~/l1_data/venv/bin/python -m unittest discover scripts/train/l1/tests

# no servidor (login node): verificação -> extração (GPU, shared mps:14) -> probe (CPU), num job
bash scripts/submit_l1_probe.sh nt2_50m ident98
bash scripts/submit_l1_probe.sh nt2_500m ident95 --mlp
```

O modelo precisa estar no cache (`HF_HOME`, unificado em `~/.cache/huggingface`) antes do job, que
roda com `HF_HUB_OFFLINE=1`. `model_tag` → id e revision pinados em `environments/config.sh`.

## Modelos

| model_tag | id | revision | pesos |
|---|---|---|---|
| `nt2_50m` | `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | `81b29e5…` | `model.safetensors` do main |
| `nt2_250m` | `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` | `8040e28…` | ver abaixo |
| `nt2_500m` | `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | `06615c1…` | `model.safetensors` do main |

**NT v2 250M:** a revision `8040e28eedfb7a9d769de536b61b265c18585f17` é o `refs/pr/3`, uma conversão
automática para safetensors feita pelo HF (SFconvertbot). Os demais arquivos são idênticos ao main
`c0f0359`. O main só tem `pytorch_model.bin`, um ZIP do `torch.save` que o storage do CISIA tende a
bloquear. No prefetch do 250M, baixar só `*.safetensors` (mais config, tokenizer e código), **nunca**
o `pytorch_model.bin`. O `meta.json` dos embeddings registra a origem dos pesos em `origem_pesos`.

NT v3 (`NTv3_650M_post`) já está pinado no `config.sh`; o adaptador `nt3` ainda não existe.

## Saídas (nada compactado)

- `$L1_EMB_DIR/<model_tag>/<ident>/`: `<split>.npy` (float16, ordem do CSV), `<split>.index.csv` e
  `meta.json` (modelo, revision, origem dos pesos, dim, camada, pooling, sha256 de CSV/.npy/index,
  seq/s, VRAM pico). Um split já extraído com mesmo modelo, revision, adaptador e CSV, e com o `.npy`
  intacto, é reaproveitado.
- `$L1_PROBE_DIR/<model_tag>/<ident>/<timestamp>/`: `report.json`, `predictions_<split>.csv`
  (dev, dev_strict, test, test_strict) e pesos do probe (`probe_logreg.json`; `probe_mlp_*.npy`
  com `--mlp`).

## Protocolo

- **Integridade antes de tudo:** sha256 dos CSVs vs `data_meta.json`, do CSV registrado na
  extração, de cada `.npy` e do index, e a ordem `window_id/orientation` linha a linha. Qualquer
  divergência aborta.
- **NT v2:** última camada (`hidden_states[-1]`), média sobre tokens com `attention_mask=1` sem
  CLS/EOS/BOS/PAD/MASK, bf16 (autocast) em GPU.
- **Probe:** padronização com média/desvio do train; logreg com `C ∈ {1e-3, …, 10}` pelo MCC do dev
  (por linha, limiar 0,5); `--mlp` com 1 camada oculta, mesmo protocolo. Cada linha (fwd e rc) é um
  exemplo.
- **Métricas** (dev, dev_strict, test, test_strict), por linha e por janela (média de fwd/rc), nos
  limiares 0,5 e de MCC máximo no dev (escolhido por nível, nunca no test):
  - total: F1 macro, MCC, AUROC, AUPRC, accuracy;
  - recall por source positiva e especificidade por source negativa;
  - l1 vs. negativos e retrovirus vs. negativos;
  - recall de l1 por faixa de `max_id_train` e de `max_id_train_local` (com n).
- O log avisa se o MCC do test_strict for ≥ ao do test (esperado: menor).

## Adicionar um modelo

Subclasse de `Adaptador` em `extract_embeddings.py` (`carregar()`, `embed(seqs)` → float32
`[n, dim]`, atributos `camada`/`pooling`/`versao`), entrada em `ADAPTADORES` e um `model_tag` no
`submit_l1_probe.sh`. O probe e as métricas não mudam.
