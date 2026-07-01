# Testes leves (propriedades de Algebra Linear), sem framework externo.
# Rodar: python testes.py

import numpy as np

import compressores as cp
import metricas
from dados import dividir_indices_estratificado


def _matriz_teste(n_tokens=16, dim=10, posto=None, semente=0):
    rng = np.random.default_rng(semente)
    if posto is not None:
        A = rng.standard_normal((n_tokens, posto))
        B = rng.standard_normal((dim, posto))
        return (A @ B.T).astype(np.float64)
    return rng.standard_normal((n_tokens, dim)).astype(np.float64)


def teste_idempotencia_sem_truncar():
    F = _matriz_teste(n_tokens=16, dim=10, semente=1)
    indices, F_hat, info = cp.comprimir_svd(F, orcamento=1.0)
    assert len(indices) == F.shape[0]
    assert F_hat.shape == F.shape
    assert info["energia_explicada"] >= 0.95


def teste_gram_psd():
    F = _matriz_teste(n_tokens=16, dim=10, semente=3)
    g = F.T @ F
    assert np.allclose(g, g.T)
    eigs = np.linalg.eigvalsh(g)
    assert np.all(eigs >= -1e-10)


def teste_valores_singulares_vs_autovalores():
    F = _matriz_teste(n_tokens=16, dim=10, semente=4)
    _, S, _ = np.linalg.svd(F, full_matrices=False)
    eigs = np.linalg.eigvalsh(F.T @ F)
    eigs = np.sort(eigs)[::-1]
    assert np.allclose(S ** 2, eigs[: len(S)], atol=1e-10)


def teste_metricas_basicas():
    assert abs(metricas.compressao(4, 2) - 0.5) < 1e-12
    R = metricas.fidelidade_reconstrucao(np.ones((4, 3)), np.ones((4, 3)))
    assert abs(R - 1.0) < 1e-12
    _, _, info = cp.comprimir_svd(_matriz_teste(n_tokens=16, dim=10, semente=1), 0.5)
    E = metricas.energia_espectral_preservada(info)
    assert E >= 0.95
    assert metricas.flops_svd_densa(64, 48) == min(64 * 48 * 48, 64 * 64 * 48)


def teste_determinismo():
    F1 = _matriz_teste(semente=7)
    F2 = _matriz_teste(semente=7)
    assert np.allclose(F1, F2)


def teste_full_mantem_todos():
    F = _matriz_teste(n_tokens=20, dim=8, semente=2)
    indices, F_hat, _ = cp.comprimir_full(F, orcamento=0.25)
    assert len(indices) == 20
    assert np.allclose(F_hat, F)


def teste_random_reprodutivel():
    F = _matriz_teste(n_tokens=30, dim=8, semente=5)
    i1, _, _ = cp.comprimir_random(F, orcamento=0.5, semente=99)
    i2, _, _ = cp.comprimir_random(F, orcamento=0.5, semente=99)
    assert np.array_equal(i1, i2)


def teste_t_minimo():
    F = _matriz_teste(n_tokens=4, dim=6, semente=1)
    indices, _, _ = cp.comprimir_svd(F, orcamento=0.01)
    assert len(indices) >= 1


def teste_svd_energia():
    F = _matriz_teste(n_tokens=32, dim=16, posto=4, semente=11)
    _, _, info_svd = cp.comprimir_svd(F, orcamento=0.5, semente=0)
    _, _, info_en = cp.comprimir_svd_energia(F, orcamento=0.5, semente=0)
    assert info_svd["pontuacoes"] is not None
    assert info_en["pontuacoes"] is not None
    assert metricas.energia_espectral_preservada(info_svd) >= 0.95


def teste_norma_zero():
    F = np.zeros((5, 4))
    assert metricas.fidelidade_reconstrucao(F, F) == 1.0
    indices, _, _ = cp.comprimir_svd(F, orcamento=0.5)
    assert len(indices) >= 1


def teste_indices_ordenados():
    F = _matriz_teste(n_tokens=25, dim=10, semente=8)
    for metodo in ("random", "svd", "svd_energia"):
        fn = cp.COMPRESSORES[metodo]
        indices, _, _ = fn(F, orcamento=0.4, semente=3)
        assert np.all(indices[:-1] <= indices[1:])


def teste_split_estratificado():
    rotulos = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    tr, te = dividir_indices_estratificado(rotulos, fracao_treino=0.5, semente=0)
    assert len(tr) == 6 and len(te) == 6
    for c in (0, 1, 2):
        assert sum(rotulos[tr] == c) == 2
        assert sum(rotulos[te] == c) == 2


def teste_epsilon_afeta_k():
    rng = np.random.default_rng(11)
    n_sv = 12
    U, _ = np.linalg.qr(rng.standard_normal((40, n_sv)))
    V, _ = np.linalg.qr(rng.standard_normal((20, n_sv)))
    singular_values = np.geomspace(10.0, 0.3, n_sv)
    F = U @ np.diag(singular_values) @ V.T
    _, _, info_baixo = cp.comprimir_svd(F, orcamento=0.5, variancia_explicada=0.90)
    _, _, info_alto = cp.comprimir_svd(F, orcamento=0.5, variancia_explicada=0.99)
    assert info_baixo["k"] < info_alto["k"]
    assert info_baixo["energia_explicada"] >= 0.90
    assert info_alto["energia_explicada"] >= 0.99


def teste_ridge_classificador():
    X = np.random.default_rng(0).standard_normal((20, 5))
    y = np.array([0] * 10 + [1] * 10)
    pred = metricas.classificador_ridge(X, y, X)
    assert metricas.acuracia(y, pred) >= 0.5


def main():
    testes = [v for k, v in globals().items() if k.startswith("teste_")]
    for t in testes:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(testes)} testes passaram.")


if __name__ == "__main__":
    main()
