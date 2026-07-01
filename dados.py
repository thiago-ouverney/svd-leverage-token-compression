# Carregamento da base IMDb (subconjunto balanceado para CPU).
#
# Requer: pip install datasets  (secao opcional do requirements.txt)

import numpy as np
from sklearn.model_selection import train_test_split

from constantes import (
    FRACAO_TREINO,
    MAX_AMOSTRAS,
    N_CLASSES_IMDB,
    N_POR_CLASSE,
    SEED,
)


def carregar_imdb(n_por_classe=N_POR_CLASSE, semente_split=SEED, max_amostras=MAX_AMOSTRAS):
    """
    Retorna textos e rotulos do subconjunto IMDb balanceado por classe.

    Returns
    -------
    textos : list[str]
    rotulos : np.ndarray[int]  (0=neg, 1=pos)
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Experimento requer 'datasets'. Instale com: pip install datasets"
        ) from exc

    ds = load_dataset("stanfordnlp/imdb", split="train")
    rng = np.random.default_rng(semente_split)

    textos, rotulos = [], []
    for classe in range(N_CLASSES_IMDB):
        indices_classe = [i for i, r in enumerate(ds["label"]) if r == classe]
        rng.shuffle(indices_classe)
        escolhidos = indices_classe[:n_por_classe]
        for idx in escolhidos:
            textos.append(ds[idx]["text"])
            rotulos.append(classe)

    if max_amostras is not None and max_amostras < len(textos):
        por_classe = max(1, max_amostras // N_CLASSES_IMDB)
        resto = max_amostras - N_CLASSES_IMDB * por_classe
        textos_out, rotulos_out = [], []
        for classe in range(N_CLASSES_IMDB):
            n_take = por_classe + (1 if classe < resto else 0)
            idx_classe = [i for i, r in enumerate(rotulos) if r == classe]
            rng.shuffle(idx_classe)
            for i in idx_classe[:n_take]:
                textos_out.append(textos[i])
                rotulos_out.append(rotulos[i])
        textos, rotulos = textos_out, rotulos_out

    return textos, np.array(rotulos, dtype=int)


def dividir_indices(n, fracao_treino=FRACAO_TREINO, semente=SEED):
    indices = np.arange(n)
    tr, te = train_test_split(
        indices, train_size=fracao_treino, random_state=semente, shuffle=True,
    )
    return tr, te


def dividir_indices_estratificado(rotulos, fracao_treino=FRACAO_TREINO, semente=SEED):
    """Split estratificado por classe."""
    indices = np.arange(len(rotulos))
    tr, te = train_test_split(
        indices,
        train_size=fracao_treino,
        random_state=semente,
        stratify=np.asarray(rotulos),
        shuffle=True,
    )
    return tr.astype(int), te.astype(int)
