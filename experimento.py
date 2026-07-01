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
    MAX_AMOSTRAS,
    MAX_TOKENS,
    METODOS_DETERMINISTICOS,
    N_POR_CLASSE,
    ORCAMENTOS,
    PASTA_RESULTADOS,
    SEED,
    SEMENTES,
)
from dados import carregar_imdb, dividir_indices_estratificado
from embeddings import texto_para_embedding
import llm_sanidade
import metricas
import visualizacao

CAMPOS_CSV = [
    "metodo", "orcamento", "semente", "modo",
    "acuracia_downstream", "energia_espectral", "fidelidade_reconstrucao",
    "compressao", "t_medio", "t_linha_medio", "custo_tempo_operador",
    "custo_flops_densa", "proxy_atencao", "acerto_exploratorio",
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


def _processar_amostras(amostras, rotulos, metodo, orcamento, idx_treino, idx_teste, semente):
    compressor = COMPRESSORES[metodo]

    vetores_treino, vetores_teste = [], []
    energias_espectrais, fidelidades, tempos, flops = [], [], [], []
    mantidos, t_originais = [], []

    idx_treino_set = set(idx_treino)
    for i, F in enumerate(amostras):
        t0 = time.perf_counter()
        indices, F_podado, info = compressor(F, orcamento, semente=semente)
        tempos.append(time.perf_counter() - t0)

        v = pooling(F_podado)
        if i in idx_treino_set:
            vetores_treino.append(v)
        else:
            vetores_teste.append(v)

        energias_espectrais.append(metricas.energia_espectral_preservada(info))
        fidelidades.append(metricas.fidelidade_reconstrucao(F, info.get("reconstrucao")))
        mantidos.append(len(indices))
        t_originais.append(F.shape[0])
        if metodo in ("svd", "svd_energia"):
            flops.append(metricas.flops_svd_densa(F.shape[0], F.shape[1]))
        else:
            flops.append(0)

    X_treino = np.vstack(vetores_treino)
    X_teste = np.vstack(vetores_teste)
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

    return {
        "acuracia_downstream": acc,
        "energia_espectral": e_esp,
        "fidelidade_reconstrucao": r_k,
        "compressao": metricas.compressao(t_medio, t_linha_medio),
        "t_medio": t_medio,
        "t_linha_medio": t_linha_medio,
        "custo_tempo_operador": float(np.mean(tempos)),
        "custo_flops_densa": float(np.mean(flops)),
        "proxy_atencao": metricas.proxy_atencao(t_medio, t_linha_medio),
        "acerto_exploratorio": metricas.acerto(acc, r_k),
    }


def rodar_grade(max_amostras=MAX_AMOSTRAS, max_tokens=MAX_TOKENS):
    linhas = []
    amostras, rotulos, textos, tempo_embed = carregar_dados(
        max_amostras=max_amostras, max_tokens=max_tokens,
    )
    idx_treino, idx_teste = dividir_indices_estratificado(rotulos, semente=SEED)
    for semente in SEMENTES:
        for metodo in COMPRESSORES:
            for orcamento in ORCAMENTOS:
                metricas_run = _processar_amostras(
                    amostras, rotulos, metodo, orcamento,
                    idx_treino, idx_teste, semente,
                )
                linhas.append({
                    "metodo": metodo,
                    "orcamento": orcamento,
                    "semente": semente,
                    "modo": "imdb",
                    **metricas_run,
                })
    return linhas


def _filtrar_para_agregacao(linhas):
    """Metodos deterministicos: uma execucao por (metodo, rho); full uma vez."""
    filtradas = []
    for l in linhas:
        if l["metodo"] == "full":
            if l["orcamento"] != ORCAMENTOS[0] or l["semente"] != 0:
                continue
        elif l["metodo"] in METODOS_DETERMINISTICOS and l["semente"] != 0:
            continue
        filtradas.append(l)
    return filtradas


def agregar(linhas):
    linhas = _filtrar_para_agregacao(linhas)
    agregado = {}
    chaves = sorted({(l["metodo"], l["orcamento"]) for l in linhas})
    for metodo, orcamento in chaves:
        grupo = [
            l for l in linhas
            if l["metodo"] == metodo and l["orcamento"] == orcamento
        ]
        accs = [g["acuracia_downstream"] for g in grupo]
        agregado.setdefault(metodo, []).append({
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
            "t_medio": float(np.mean([g["t_medio"] for g in grupo])),
            "t_linha_medio": float(np.mean([g["t_linha_medio"] for g in grupo])),
            "custo_tempo_media": float(np.mean([g["custo_tempo_operador"] for g in grupo])),
            "proxy_atencao": float(np.mean([g["proxy_atencao"] for g in grupo])),
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

    linhas = rodar_grade(
        max_amostras=args.max_amostras,
        max_tokens=args.max_tokens,
    )
    caminho_csv = os.path.join(PASTA_RESULTADOS, "resultados_imdb.csv")
    salvar_csv(linhas, caminho_csv)
    salvar_csv(linhas, os.path.join(PASTA_RESULTADOS, "resultados.csv"))

    agregado = agregar(linhas)
    visualizacao.salvar_grafico_tradeoff(
        agregado, os.path.join(PASTA_RESULTADOS, "tradeoff_acerto_compressao.png")
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

    print("Resumo (media de acuracia por metodo e orcamento):")
    for metodo, pontos in sorted(agregado.items()):
        print("  [{}]".format(metodo))
        for p in sorted(pontos, key=lambda x: x["compressao"]):
            print(
                "    rho={:.3f} | C={:.3f} | acc={:.3f}+/-{:.3f}".format(
                    p["orcamento"], p["compressao"], p["acuracia_media"],
                    p["acuracia_desvio"],
                )
            )

    print("\nAmbiente:", ambiente)
    print("CSV:", os.path.join(PASTA_RESULTADOS, "resultados.csv"))


if __name__ == "__main__":
    main()
