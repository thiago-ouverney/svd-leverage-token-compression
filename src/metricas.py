# Metricas do trabalho: acuracia downstream, energia espectral, fidelidade, compressao e custo.
#
# Classificador ridge (sklearn): treina e testa com embeddings comprimidos
# pelo mesmo metodo e orcamento; o metodo "full" serve como teto de referencia.
#
# E_k^energia = sum_{j<=k} sigma_j^2 / sum_j sigma_j^2  (subespaco SVD)
# R_k = 1 - ||F - F_k||_F / ||F||_F  (fidelidade de reconstrucao, nao poda S)
# Compressao = 1 - T'/T

import numpy as np
from sklearn.linear_model import RidgeClassifier

from constantes import RIDGE_LAMBDA


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


def classificador_ridge(X_treino, y_treino, X_teste, lam=RIDGE_LAMBDA):
    """Ridge binario/multi-classe via sklearn (alpha=lam, intercepto nao regularizado)."""
    clf = RidgeClassifier(alpha=lam, fit_intercept=True)
    clf.fit(X_treino, y_treino)
    return clf.predict(X_teste)


# Alias usado pelo experimento
classificador_linear = classificador_ridge


def acuracia(y_verdadeiro, y_previsto):
    return float(np.mean(np.asarray(y_verdadeiro) == np.asarray(y_previsto)))
