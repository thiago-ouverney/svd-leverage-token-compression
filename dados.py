# Carregamento da base IMDb (subconjunto balanceado para CPU).
#
# Requer: pip install datasets  (secao opcional do requirements.txt)

import numpy as np

N_CLASSES_IMDB = 2


def carregar_imdb(n_por_classe=200, semente_split=42, max_amostras=None):
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


def dividir_indices(n, fracao_treino=0.7, semente=42):
    rng = np.random.default_rng(semente)
    perm = rng.permutation(n)
    corte = int(round(fracao_treino * n))
    return perm[:corte], perm[corte:]


def dividir_indices_estratificado(rotulos, fracao_treino=0.7, semente=42):
    """Split 70/30 estratificado por classe."""
    rng = np.random.default_rng(semente)
    rotulos = np.asarray(rotulos)
    idx_treino, idx_teste = [], []
    for classe in np.unique(rotulos):
        idx_classe = np.where(rotulos == classe)[0]
        rng.shuffle(idx_classe)
        corte = int(round(fracao_treino * len(idx_classe)))
        corte = max(1, min(corte, len(idx_classe) - 1)) if len(idx_classe) > 1 else 1
        idx_treino.extend(idx_classe[:corte])
        idx_teste.extend(idx_classe[corte:])
    return np.array(idx_treino, dtype=int), np.array(idx_teste, dtype=int)
