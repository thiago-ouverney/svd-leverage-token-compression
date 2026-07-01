# Figuras: trade-off Acuracia x Compressao e custo por metodo.

import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


CORES = {
    "full": "#2ca02c",
    "random": "#9467bd",
    "norm": "#ff7f0e",
    "first": "#8c564b",
    "svd": "#1f77b4",
    "svd_energia": "#d62728",
}

ROTULOS = {
    "full": "Full",
    "random": "Random",
    "norm": "Norm",
    "first": "First",
    "svd": "SVD-Leverage",
    "svd_energia": "SVD-Energia",
}


def salvar_grafico_tradeoff(agregado, caminho):
    plt.figure(figsize=(8, 5))
    for metodo, pontos in sorted(agregado.items()):
        pts = sorted(pontos, key=lambda p: p["compressao"])
        x = [p["compressao"] for p in pts]
        y = [p["acuracia_media"] for p in pts]
        e = [p["acuracia_desvio"] for p in pts]
        plt.errorbar(
            x, y, yerr=e, marker="o", capsize=3,
            label=ROTULOS.get(metodo, metodo),
            color=CORES.get(metodo, "#333333"),
        )
    plt.xlabel("Compressao (1 - T'/T)")
    plt.ylabel("Acuracia downstream")
    plt.title("Trade-off Acuracia x Compressao (baselines + SVD)")
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
    return caminho


def salvar_grafico_custo(agregado, caminho):
    metodos_svd = [m for m in ("svd", "svd_energia", "full", "random", "norm", "first")
                   if m in agregado]
    rho_ref = 0.5
    tempos = []
    labels = []
    cores = []
    for metodo in metodos_svd:
        pts = {p["orcamento"]: p for p in agregado[metodo]}
        if rho_ref in pts:
            tempos.append(pts[rho_ref]["custo_tempo_media"])
            labels.append(ROTULOS.get(metodo, metodo))
            cores.append(CORES.get(metodo, "#333333"))

    plt.figure(figsize=(7, 4))
    plt.bar(labels, tempos, color=cores)
    plt.ylabel("Tempo medio do operador (s)")
    plt.title("Custo do operador por metodo (rho=0.5)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
    return caminho
