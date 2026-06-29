# Metricas do trabalho: acuracia downstream, energia espectral, fidelidade, compressao e custo.
#
# Classificador ridge: treina e testa com embeddings comprimidos pelo
# mesmo metodo e orcamento; o metodo "full" serve como teto de referencia.
#
# E_k^energia = sum_{j<=k} sigma_j^2 / sum_j sigma_j^2  (subespaco SVD)
# R_k = 1 - ||F - F_k||_F / ||F||_F  (fidelidade de reconstrucao, nao poda S)
# Compressao = 1 - T'/T

import numpy as np

RIDGE_LAMBDA = 1.0


def compressao(n_tokens, n_mantidos):
    return 1.0 - (n_mantidos / n_tokens)


def energia_espectral_preservada(info):
    """Fracao de energia espectral capturada pelo subespaco de posto k."""
    if info is None or info.get("energia_explicada") is None:
        return float("nan")
    return float(info["energia_explicada"])


def fidelidade_reconstrucao(F, F_reconstruida):
    """Fidelidade relativa R_k da aproximacao F_k (nao mede qualidade de S)."""
    if F_reconstruida is None:
        return float("nan")
    erro = np.linalg.norm(F - F_reconstruida)
    base = np.linalg.norm(F)
    if base == 0.0:
        return 1.0
    return float(1.0 - erro / base)


# Alias retrocompativel
energia_preservada = fidelidade_reconstrucao


def acerto(acuracia_downstream, fidelidade, alpha=0.5):
    """Indice exploratorio agregado; nao e a metrica principal do relatorio."""
    if np.isnan(fidelidade):
        return float(acuracia_downstream)
    return float(alpha * acuracia_downstream + (1.0 - alpha) * fidelidade)


def flops_svd_densa(n_tokens, dim):
    """Custo de numpy.linalg.svd (decomposicao densa completa)."""
    return int(min(n_tokens * dim * dim, n_tokens * n_tokens * dim))


def flops_estimados_truncado(n_tokens, dim, k):
    """Referencia conceitual para SVD truncada/randomizada (nao usada no codigo)."""
    return int(n_tokens * dim * max(1, k))


def proxy_atencao(n_tokens, n_mantidos):
    """Proxy qualitativo de reducao de custo em atencao densa O(T^2) no prefill."""
    if n_tokens <= 0:
        return 0.0
    return float(1.0 - (n_mantidos / n_tokens) ** 2)


def classificador_ridge(X_treino, y_treino, X_teste, lam=RIDGE_LAMBDA):
    """
    Ridge multi-classe com regularizacao fixa (lambda=1).
    W = (X^T X + lam I)^{-1} X^T Y, com bias via coluna de uns.
    """
    classes = np.unique(y_treino)
    Y = np.zeros((len(y_treino), len(classes)))
    for j, c in enumerate(classes):
        Y[y_treino == c, j] = 1.0

    A = np.hstack([X_treino, np.ones((X_treino.shape[0], 1))])
    n_features = A.shape[1]
    reg = lam * np.eye(n_features)
    reg[-1, -1] = 0.0  # nao regulariza bias
    W = np.linalg.solve(A.T @ A + reg, A.T @ Y)

    A_teste = np.hstack([X_teste, np.ones((X_teste.shape[0], 1))])
    escores = A_teste @ W
    return classes[np.argmax(escores, axis=1)]


# Alias usado pelo experimento
classificador_linear = classificador_ridge


def acuracia(y_verdadeiro, y_previsto):
    return float(np.mean(np.asarray(y_verdadeiro) == np.asarray(y_previsto)))
