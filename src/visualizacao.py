# Figuras: trade-off Acuracia x Compressao e epsilon x k.

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from compressores import _svd_decomposicao
from constantes import EPSILONS


CORES = {
    "full": "#2ca02c",
    "random": "#9467bd",
    "svd": "#1f77b4",
    "svd_energia": "#d62728",
}

ROTULOS = {
    "full": "Full",
    "random": "Random",
    "svd": "SVD-Leverage",
    "svd_energia": "SVD-Energia",
}

CORES_EPS = {
    0.90: "#ff7f0e",
    0.95: "#1f77b4",
    0.99: "#d62728",
}


def coletar_cortes_epsilon(amostras, epsilons):
    """Curva de energia acumulada (review representativa) e k medio por epsilon."""
    t_medio = float(np.mean([F.shape[0] for F in amostras]))
    idx_repr = int(np.argmin([abs(F.shape[0] - t_medio) for F in amostras]))
    F_repr = amostras[idx_repr]

    _, S, _, _, _ = _svd_decomposicao(F_repr, variancia_explicada=1.0)
    energia = S ** 2
    total = float(np.sum(energia))
    acumulada = np.cumsum(energia) / total if total > 0 else np.zeros_like(energia)

    cortes_repr = {}
    k_medio = {}
    for eps in epsilons:
        ks = []
        for F in amostras:
            _, _, _, k, _ = _svd_decomposicao(F, variancia_explicada=eps)
            ks.append(k)
        k_medio[eps] = float(np.mean(ks))
        _, _, _, k_repr, _ = _svd_decomposicao(F_repr, variancia_explicada=eps)
        cortes_repr[eps] = k_repr

    return acumulada, cortes_repr, k_medio, idx_repr


def salvar_grafico_epsilon_k(amostras, epsilons, caminho):
    acumulada, cortes_repr, k_medio, idx_repr = coletar_cortes_epsilon(amostras, epsilons)
    modos = np.arange(1, len(acumulada) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(modos, acumulada, color="#333333", linewidth=1.5, label="Energia acumulada")
    ax1.set_xlabel("Indice do modo singular j")
    ax1.set_ylabel("Energia espectral acumulada")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)

    for eps in epsilons:
        k = cortes_repr[eps]
        cor = CORES_EPS.get(eps, "#333333")
        ax1.axvline(k, color=cor, linestyle="--", linewidth=1.2, alpha=0.85)
        ax1.axhline(eps, color=cor, linestyle=":", linewidth=0.9, alpha=0.5)
        ax1.scatter(
            [k], [acumulada[k - 1]], color=cor, s=60, zorder=5,
            label=f"eps={eps:.2f}, k={k}, k_medio={k_medio[eps]:.1f}",
        )

    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title(
        f"Cortes k(epsilon) na curva espectral (review representativa, idx={idx_repr})"
    )
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
    return caminho


def salvar_grafico_tradeoff(agregado, caminho, epsilons=None):
    epsilons = sorted(epsilons or EPSILONS, reverse=True)
    n_eps = len(epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(5 * n_eps, 4.5), sharey=True)
    if n_eps == 1:
        axes = [axes]

    metodos_ordem = ["full", "random", "svd", "svd_energia"]

    for ax, eps in zip(axes, epsilons):
        for metodo in metodos_ordem:
            if metodo not in agregado:
                continue
            pts = sorted(
                [p for p in agregado[metodo] if p["epsilon"] == eps],
                key=lambda p: p["compressao"],
            )
            if not pts:
                continue
            x = [p["compressao"] for p in pts]
            y = [p["acuracia_media"] for p in pts]
            if metodo == "full":
                ax.axhline(y[0], color=CORES[metodo], linestyle="--", linewidth=1.2,
                           label=ROTULOS[metodo], alpha=0.8)
            else:
                ax.plot(
                    x, y, marker="o",
                    label=ROTULOS.get(metodo, metodo),
                    color=CORES.get(metodo, "#333333"),
                )
        ax.set_xlabel("Compressao (1 - T'/T)")
        ax.set_title(f"epsilon = {eps:.2f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    axes[0].set_ylabel("Acuracia downstream")
    fig.suptitle("Trade-off Acuracia x Compressao por epsilon", y=1.02)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
    return caminho
