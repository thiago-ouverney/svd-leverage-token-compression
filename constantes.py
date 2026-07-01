# Parametros centrais do experimento (seeds via variaveis de ambiente).

import os

_DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))


def _int_env(nome: str, padrao: int) -> int:
    return int(os.environ.get(nome, padrao))


# Seeds (variaveis de ambiente)
SEED = _int_env("SEED", 42)
SEMENTES = [int(s) for s in os.environ.get("SEMENTES", "0,1,2").split(",")]

# Dataset IMDb
N_CLASSES_IMDB = 2
N_POR_CLASSE = 200
MAX_AMOSTRAS = 400
MAX_TOKENS = 256
FRACAO_TREINO = 0.7

# Experimento
ORCAMENTOS = [0.75, 0.5, 0.25, 0.125]
PASTA_RESULTADOS = os.path.join(_DIR_PROJETO, "resultados")
METODOS_DETERMINISTICOS = {"full", "norm", "first", "svd", "svd_energia"}

# Metricas / compressores
RIDGE_LAMBDA = 1.0
TOL_SINGULAR = 1e-12
