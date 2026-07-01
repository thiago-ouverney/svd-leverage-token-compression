# Parametros centrais do experimento.

import os

_DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _int_env(nome: str, padrao: int) -> int:
    return int(os.environ.get(nome, padrao))


# Split treino/teste (variavel de ambiente)
SEED = _int_env("SEED", 42)

# Dataset IMDb
N_CLASSES_IMDB = 2
N_POR_CLASSE = 200
MAX_AMOSTRAS = 400
MAX_TOKENS = 256
FRACAO_TREINO = 0.7

# Experimento
EPSILONS = [0.90, 0.95, 0.99]
ORCAMENTOS = [0.50, 0.25, 0.125]
PASTA_RESULTADOS = os.path.join(_DIR_RAIZ, "resultados")

# Metricas / compressores
RIDGE_LAMBDA = 1.0
TOL_SINGULAR = 1e-12
