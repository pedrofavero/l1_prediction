"""Metricas do baseline L1 (numpy + scikit-learn), funcoes puras.

Um `conjunto` e um dict de arrays numpy alinhados por linha: window_id, label,
source, max_id_train, max_id_train_local (NaN quando ausente) e, por linha,
orientation. `prob` = P(classe 1). Predicao positiva quando prob >= limiar.
"""
import numpy as np

FAIXAS_ID = (("<0.80", 0.80), ("0.80-0.90", 0.90), ("0.90-0.95", 0.95), (">=0.95", None))
POSITIVAS = ("l1", "retrovirus")
NEGATIVAS = ("te", "background", "markov")


def r4(x):
    return None if x is None else round(float(x), 4)


def binarias(y, prob, limiar):
    """F1 macro, MCC, AUROC, AUPRC (average precision) e accuracy. AUROC/AUPRC = None com uma classe so."""
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                                 roc_auc_score)

    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= limiar).astype(int)
    duas = len(np.unique(y)) == 2
    return {"n": int(len(y)), "n_pos": int(y.sum()), "limiar": r4(limiar),
            "accuracy": r4(accuracy_score(y, pred)) if len(y) else None,
            "f1_macro": r4(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else None,
            "mcc": r4(matthews_corrcoef(y, pred)) if len(y) else None,
            "auroc": r4(roc_auc_score(y, prob)) if duas else None,
            "auprc": r4(average_precision_score(y, prob)) if duas else None}


def limiar_mcc(y, prob):
    """Limiar (entre os valores de prob) que maximiza o MCC -> (limiar, mcc). Empate: o mais perto de 0,5."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(prob, dtype=float)
    ordem = np.argsort(-p, kind="mergesort")
    ps, ys = p[ordem], y[ordem]
    ultimos = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]  # ultimo indice de cada valor de prob
    tp = np.cumsum(ys)[ultimos].astype(float)
    fp = np.cumsum(1 - ys)[ultimos].astype(float)
    pos, neg = float(ys.sum()), float(len(ys) - ys.sum())
    fn, tn = pos - tp, neg - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    limiares = ps[ultimos]
    candidatos = np.nonzero(mcc >= mcc.max() - 1e-12)[0]
    i = candidatos[np.argmin(np.abs(limiares[candidatos] - 0.5))]
    return float(limiares[i]), float(mcc[i])


def por_janela(conj, prob):
    """Media das probabilidades das linhas (fwd/rc) de cada window_id -> (conjunto por janela, prob).

    Ordem: primeira aparicao. ValueError se linhas da mesma janela tiverem rotulos diferentes.
    """
    wid = conj["window_id"]
    unicos, primeira, inv = np.unique(wid, return_index=True, return_inverse=True)
    soma = np.bincount(inv, weights=np.asarray(prob, dtype=float))
    cont = np.bincount(inv)
    y = np.asarray(conj["label"])
    rot_min = np.full(len(unicos), 2)
    rot_max = np.full(len(unicos), -1)
    np.minimum.at(rot_min, inv, y)
    np.maximum.at(rot_max, inv, y)
    if (rot_min != rot_max).any():
        raise ValueError(f"{int((rot_min != rot_max).sum())} janela(s) com rotulos diferentes entre fwd e rc")
    ordem = np.argsort(primeira, kind="mergesort")
    idx = primeira[ordem]
    janelas = {k: np.asarray(v)[idx] for k, v in conj.items() if k != "orientation"}
    return janelas, (soma / cont)[ordem]


def por_source(source, y, pred):
    """Recall por source positiva e especificidade por source negativa."""
    out = {}
    for src in POSITIVAS + NEGATIVAS:
        m = source == src
        if not m.any():
            continue
        if src in POSITIVAS:
            out[src] = {"n": int(m.sum()), "recall": r4((pred[m] == 1).mean())}
        else:
            out[src] = {"n": int(m.sum()), "especificidade": r4((pred[m] == 0).mean())}
    return out


def alvo_vs_negativos(source, y, prob, alvo, limiar):
    """Binarias so com as linhas de `alvo` e todos os negativos (ex.: l1 vs. negativos). None sem `alvo`."""
    m = (source == alvo) | (np.asarray(y) == 0)
    if not (source == alvo).any():
        return None
    return binarias(np.asarray(y)[m], np.asarray(prob)[m], limiar)


def faixa_id(x):
    for nome, limite in FAIXAS_ID:
        if limite is None or x < limite:
            return nome


def recall_por_faixa(max_id, source, y, pred):
    """Recall dos positivos l1 por faixa de identidade com o train (n de cada faixa; NaN conta a parte)."""
    m = (source == "l1") & (np.asarray(y) == 1)
    ids, pr = np.asarray(max_id, dtype=float)[m], pred[m]
    out = {}
    sem = np.isnan(ids)
    for nome, _ in FAIXAS_ID:
        sel = np.array([not s and faixa_id(x) == nome for x, s in zip(ids, sem)], dtype=bool)
        n = int(sel.sum())
        out[nome] = {"n": n, "recall": r4(pr[sel].mean()) if n else None}
    if sem.any():
        out["sem_valor"] = {"n": int(sem.sum())}
    return out


def avaliar(conj, prob, limiar):
    """Todas as metricas de um conjunto num limiar."""
    y = np.asarray(conj["label"], dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= limiar).astype(int)
    src = np.asarray(conj["source"])
    return {"total": binarias(y, prob, limiar),
            "por_source": por_source(src, y, pred),
            "l1_vs_negativos": alvo_vs_negativos(src, y, prob, "l1", limiar),
            "retrovirus_vs_negativos": alvo_vs_negativos(src, y, prob, "retrovirus", limiar),
            "faixas_max_id_train": recall_por_faixa(conj["max_id_train"], src, y, pred),
            "faixas_max_id_train_local": recall_por_faixa(conj["max_id_train_local"], src, y, pred)}


def limiares_do_dev(conj_dev, prob_dev):
    """{"linha": t, "janela": t}: limiar de MCC maximo no dev, por nivel. Nunca escolher no test."""
    jan, pj = por_janela(conj_dev, prob_dev)
    return {"linha": limiar_mcc(conj_dev["label"], prob_dev)[0], "janela": limiar_mcc(jan["label"], pj)[0]}


def avaliar_completo(conj, prob, limiares_dev):
    """{nivel: {"0.5" | "dev": avaliar(...)}} para nivel linha e janela."""
    jan, pj = por_janela(conj, prob)
    out = {}
    for nivel, (c, p) in (("linha", (conj, prob)), ("janela", (jan, pj))):
        out[nivel] = {"0.5": avaliar(c, p, 0.5), "dev": avaliar(c, p, limiares_dev[nivel])}
    return out
