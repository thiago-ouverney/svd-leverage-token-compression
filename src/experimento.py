# Experimento: compressao de tokens via SVD + leverage scores e baselines (IMDb).

import argparse
import csv
import os
import platform
import sys
import time

import numpy as np

from compressores import COMPRESSORES
from constantes import (
    EPSILONS,
    MAX_AMOSTRAS,
    MAX_TOKENS,
    N_POR_CLASSE,
    ORCAMENTOS,
    PASTA_RESULTADOS,
    SEED,
)
from dados import carregar_imdb, dividir_indices_estratificado
from embeddings import texto_para_embedding
import llm_sanidade
import metricas
import visualizacao

CAMPOS_CSV = [
    "metodo", "epsilon", "orcamento", "semente", "modo",
    "acuracia_downstream", "energia_espectral", "fidelidade_reconstrucao",
    "compressao", "t_medio", "t_linha_medio", "k_medio", "custo_tempo_operador",
]


def pooling(F):
    v = F.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def carregar_dados(n_por_classe=N_POR_CLASSE, max_amostras=MAX_AMOSTRAS, max_tokens=MAX_TOKENS):
    textos, rotulos = carregar_imdb(
        n_por_classe=n_por_classe, max_amostras=max_amostras,
    )
    amostras = []
    t0_embed = time.perf_counter()
    for texto in textos:
        amostras.append(texto_para_embedding(texto, max_tokens=max_tokens))
    tempo_embedding_total = time.perf_counter() - t0_embed
    tempo_embedding_medio = tempo_embedding_total / max(1, len(textos))
    return amostras, rotulos, textos, tempo_embedding_medio


def _processar_amostras(
    amostras, rotulos, metodo, orcamento, idx_treino, idx_teste, semente,
    variancia_explicada=0.95,
):
    compressor = COMPRESSORES[metodo]

    vetores = [None] * len(amostras)
    energias_espectrais, fidelidades, tempos = [], [], []
    mantidos, t_originais, ks = [], [], []

    for i, F in enumerate(amostras):
        t0 = time.perf_counter()
        indices, F_podado, info = compressor(
            F, orcamento, semente=semente, variancia_explicada=variancia_explicada,
        )
        tempos.append(time.perf_counter() - t0)

        vetores[i] = pooling(F_podado)

        energias_espectrais.append(metricas.energia_espectral_preservada(info))
        fidelidades.append(metricas.fidelidade_reconstrucao(F, info.get("reconstrucao")))
        mantidos.append(len(indices))
        t_originais.append(F.shape[0])
        if info.get("k") is not None:
            ks.append(info["k"])

    X_treino = np.vstack([vetores[i] for i in idx_treino])
    X_teste = np.vstack([vetores[i] for i in idx_teste])
    y_treino = rotulos[idx_treino]
    y_teste = rotulos[idx_teste]

    previsto = metricas.classificador_ridge(X_treino, y_treino, X_teste)
    acc = metricas.acuracia(y_teste, previsto)

    def _media_valida(vals):
        ok = [v for v in vals if not np.isnan(v)]
        return float(np.mean(ok)) if ok else float("nan")

    e_esp = _media_valida(energias_espectrais)
    r_k = _media_valida(fidelidades)
    t_medio = float(np.mean(t_originais))
    t_linha_medio = float(np.mean(mantidos))
    k_medio = float(np.mean(ks)) if ks else float("nan")

    return {
        "acuracia_downstream": acc,
        "energia_espectral": e_esp,
        "fidelidade_reconstrucao": r_k,
        "compressao": metricas.compressao(t_medio, t_linha_medio),
        "t_medio": t_medio,
        "t_linha_medio": t_linha_medio,
        "k_medio": k_medio,
        "custo_tempo_operador": float(np.mean(tempos)),
    }


def _combos_por_metodo(metodo):
    if metodo == "full":
        return [(EPSILONS[0], ORCAMENTOS[0])]
    if metodo == "random":
        return [(EPSILONS[0], rho) for rho in ORCAMENTOS]
    return [(eps, rho) for eps in EPSILONS for rho in ORCAMENTOS]


def expandir_grade(linhas):
    """Replica full/random em todas as combinacoes (epsilon, orcamento) -> 36 linhas."""
    por_chave = {
        (l["metodo"], l["epsilon"], l["orcamento"]): l
        for l in linhas
    }
    ref_full = por_chave[("full", EPSILONS[0], ORCAMENTOS[0])]
    expandidas = []
    for metodo in COMPRESSORES:
        for epsilon in EPSILONS:
            for orcamento in ORCAMENTOS:
                if metodo == "full":
                    linha = {**ref_full, "epsilon": epsilon, "orcamento": orcamento}
                elif metodo == "random":
                    ref = por_chave[("random", EPSILONS[0], orcamento)]
                    linha = {**ref, "epsilon": epsilon}
                else:
                    linha = por_chave[(metodo, epsilon, orcamento)]
                expandidas.append(linha)
    return expandidas


def rodar_grade(
    max_amostras=MAX_AMOSTRAS,
    max_tokens=MAX_TOKENS,
    amostras=None,
    rotulos=None,
):
    linhas = []
    if amostras is None:
        amostras, rotulos, _, _ = carregar_dados(
            max_amostras=max_amostras, max_tokens=max_tokens,
        )
    idx_treino, idx_teste = dividir_indices_estratificado(rotulos, semente=SEED)
    for metodo in COMPRESSORES:
        for epsilon, orcamento in _combos_por_metodo(metodo):
            metricas_run = _processar_amostras(
                amostras, rotulos, metodo, orcamento,
                idx_treino, idx_teste, semente=0,
                variancia_explicada=epsilon,
            )
            linhas.append({
                "metodo": metodo,
                "epsilon": epsilon,
                "orcamento": orcamento,
                "semente": 0,
                "modo": "imdb",
                **metricas_run,
            })
    return expandir_grade(linhas)


def _filtrar_para_agregacao(linhas):
    return linhas


def agregar(linhas):
    linhas = _filtrar_para_agregacao(linhas)
    agregado = {}
    chaves = sorted({
        (l["metodo"], l["orcamento"], l["epsilon"]) for l in linhas
    })
    for metodo, orcamento, epsilon in chaves:
        grupo = [
            l for l in linhas
            if l["metodo"] == metodo
            and l["orcamento"] == orcamento
            and l["epsilon"] == epsilon
        ]
        accs = [g["acuracia_downstream"] for g in grupo]
        agregado.setdefault(metodo, []).append({
            "epsilon": epsilon,
            "orcamento": orcamento,
            "compressao": float(np.mean([g["compressao"] for g in grupo])),
            "acuracia_media": float(np.mean(accs)),
            "acuracia_desvio": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "energia_espectral_media": float(np.mean([
                g["energia_espectral"] for g in grupo if not np.isnan(g["energia_espectral"])
            ])) if any(not np.isnan(g["energia_espectral"]) for g in grupo) else float("nan"),
            "fidelidade_media": float(np.mean([
                g["fidelidade_reconstrucao"] for g in grupo
                if not np.isnan(g["fidelidade_reconstrucao"])
            ])) if any(not np.isnan(g["fidelidade_reconstrucao"]) for g in grupo) else float("nan"),
            "k_medio": float(np.mean([
                g["k_medio"] for g in grupo if not np.isnan(g["k_medio"])
            ])) if any(not np.isnan(g["k_medio"]) for g in grupo) else float("nan"),
            "t_medio": float(np.mean([g["t_medio"] for g in grupo])),
            "t_linha_medio": float(np.mean([g["t_linha_medio"] for g in grupo])),
            "custo_tempo_media": float(np.mean([g["custo_tempo_operador"] for g in grupo])),
        })
    return agregado


def salvar_csv(linhas, caminho):
    with open(caminho, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV, extrasaction="ignore")
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow({c: linha.get(c, "") for c in CAMPOS_CSV})


def registrar_ambiente():
    return {
        "dataset": "imdb",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "plataforma": platform.platform(),
        "processador": platform.processor() or "desconhecido",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compressao de tokens via SVD + leverage scores (IMDb)"
    )
    parser.add_argument("--max-amostras", type=int, default=MAX_AMOSTRAS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--sanidade-gpt2", action="store_true")
    args = parser.parse_args()

    print("Trabalho Final ALC - Compressao de tokens (SVD + baselines)")
    print("Dataset: IMDb (max_tokens={})\n".format(args.max_tokens))

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    t0_total = time.perf_counter()

    amostras, rotulos, _, _ = carregar_dados(
        max_amostras=args.max_amostras,
        max_tokens=args.max_tokens,
    )
    linhas = rodar_grade(
        max_amostras=args.max_amostras,
        max_tokens=args.max_tokens,
        amostras=amostras,
        rotulos=rotulos,
    )
    caminho_csv = os.path.join(PASTA_RESULTADOS, "resultados_imdb.csv")
    salvar_csv(linhas, caminho_csv)
    salvar_csv(linhas, os.path.join(PASTA_RESULTADOS, "resultados.csv"))

    agregado = agregar(linhas)
    visualizacao.salvar_grafico_tradeoff(
        agregado, os.path.join(PASTA_RESULTADOS, "tradeoff_acerto_compressao.png"),
    )
    visualizacao.salvar_grafico_epsilon_k(
        amostras, EPSILONS,
        os.path.join(PASTA_RESULTADOS, "epsilon_k_cortes.png"),
    )

    tempo_total = time.perf_counter() - t0_total
    ambiente = registrar_ambiente()
    ambiente["tempo_total_s"] = tempo_total
    ambiente["max_tokens"] = args.max_tokens
    ambiente["max_amostras"] = args.max_amostras

    if args.sanidade_gpt2:
        textos, _ = carregar_imdb(n_por_classe=5, max_amostras=5)
        scores = llm_sanidade.executar_sanidade(textos, usar_gpt2=True, n_amostras=5)
        print("Sanidade gpt2 (5 amostras):", scores)

    print("Resumo (media de acuracia por metodo, epsilon e orcamento):")
    for metodo, pontos in sorted(agregado.items()):
        print("  [{}]".format(metodo))
        for p in sorted(pontos, key=lambda x: (x["epsilon"], x["compressao"])):
            print(
                "    eps={:.2f} rho={:.3f} | C={:.3f} | acc={:.3f}+/-{:.3f} | k={}".format(
                    p["epsilon"], p["orcamento"], p["compressao"], p["acuracia_media"],
                    p["acuracia_desvio"],
                    f"{p['k_medio']:.1f}" if not np.isnan(p["k_medio"]) else "nan",
                )
            )

    print("\nAmbiente:", ambiente)
    print("CSV:", os.path.join(PASTA_RESULTADOS, "resultados.csv"))


if __name__ == "__main__":
    main()
