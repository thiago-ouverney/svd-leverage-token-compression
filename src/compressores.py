# Compressores de tokens: baselines + SVD + leverage scores (SVD-Prune textual).
#
# Objeto: F in R^{T x D} (uma linha por token).
# Retorna (indices_mantidos, F[indices], info).

import numpy as np

from constantes import TOL_SINGULAR


def _tokens_a_manter(n_tokens, orcamento):
    if 0 < orcamento <= 1:
        return max(1, int(round(orcamento * n_tokens)))
    return max(1, int(orcamento))


def selecionar_sob_orcamento(pontuacoes, n_manter):
    n_manter = int(max(1, min(n_manter, len(pontuacoes))))
    indices = np.argsort(pontuacoes)[::-1][:n_manter]
    return np.sort(indices)


def _info_base(k=None, reconstrucao=None, pontuacoes=None, energia_explicada=None):
    return {
        "k": k,
        "energia_explicada": energia_explicada,
        "reconstrucao": reconstrucao,
        "pontuacoes": pontuacoes,
    }


def _svd_decomposicao(F, variancia_explicada=0.95):
    U, S, Vt = np.linalg.svd(F, full_matrices=False) # SVD economica
    # Zerando valores singulares menores que TOL_SINGULAR
    S = np.where(S < TOL_SINGULAR, 0.0, S)
    # Energia espectral
    energia = S ** 2
    total = np.sum(energia)
    if total <= TOL_SINGULAR:
        k = 1
        acumulada = np.array([0.0]) # Caso degradado
    else:
        # Acha o k para a variancia explicada
        acumulada = np.cumsum(energia) / total
        k = int(np.searchsorted(acumulada, variancia_explicada) + 1)
        k = max(1, min(k, len(S)))
    return U, S, Vt, k, float(acumulada[k - 1] if len(acumulada) else 0.0)


def pontuar_leverage(U, k):
    uk = U[:, :k]
    return (uk ** 2).sum(axis=1) / k


def pontuar_leverage_energia(U, S, k):
    uk = U[:, :k]
    sk = S[:k]
    return ((uk ** 2) * (sk ** 2)).sum(axis=1)


def comprimir_full(F, orcamento=1.0, semente=0, **_):
    del orcamento, semente
    indices = np.arange(F.shape[0])
    return indices, F.copy(), _info_base(reconstrucao=None)


def comprimir_random(F, orcamento, semente=0, **_):
    n_manter = _tokens_a_manter(F.shape[0], orcamento)
    rng = np.random.default_rng(semente)
    indices = np.sort(rng.choice(F.shape[0], size=n_manter, replace=False))
    return indices, F[indices], _info_base(reconstrucao=None)


def _comprimir_svd_core(F, orcamento, variancia_explicada, pontuar_fn, semente=0):
    del semente
    n_manter = _tokens_a_manter(F.shape[0], orcamento)
    U, S, Vt, k, energia_explicada = _svd_decomposicao(F, variancia_explicada)
    pontuacoes = pontuar_fn(U, S, k)
    indices = selecionar_sob_orcamento(pontuacoes, n_manter)
    reconstrucao = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    info = _info_base(
        k=k,
        energia_explicada=energia_explicada,
        reconstrucao=reconstrucao,
        pontuacoes=pontuacoes,
    )
    return indices, F[indices], info


def comprimir_svd(F, orcamento, semente=0, variancia_explicada=0.95, **_):
    return _comprimir_svd_core(
        F, orcamento, variancia_explicada,
        lambda U, S, k: pontuar_leverage(U, k),
        semente=semente,
    )


def comprimir_svd_energia(F, orcamento, semente=0, variancia_explicada=0.95, **_):
    return _comprimir_svd_core(
        F, orcamento, variancia_explicada,
        pontuar_leverage_energia,
        semente=semente,
    )


COMPRESSORES = {
    "full": comprimir_full,
    "random": comprimir_random,
    "svd": comprimir_svd,
    "svd_energia": comprimir_svd_energia,
}
