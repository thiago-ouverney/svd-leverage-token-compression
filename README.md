# SVD + leverage scores — compressão de tokens

Trabalho Final de **Álgebra Linear Computacional** (UFF 2026.1): compressão token-level de embeddings via SVD truncada e leverage scores, com baselines e avaliação downstream em IMDb.

**Repositório:** [github.com/thiago-ouverney/svd-leverage-token-compression](https://github.com/thiago-ouverney/svd-leverage-token-compression)

## Pipeline

```
review IMDb → F (T×D) → poda (4 operadores) → Ridge → métricas
```

Operadores de poda:

- `full` — sem poda (teto de referência)
- `random` — tokens aleatórios (semente fixa 0)
- `svd` — leverage scores ℓᵢ/k (SVD truncada a 95% de energia)
- `svd_energia` — leverage ponderado por Σ

## Instalação

```bash
make setup        # cria .venv, instala deps (pyproject.toml) e kernel Jupyter
source .venv/bin/activate
make experiment   # grade IMDb (~160 s CPU)
```

Na primeira execução, o HuggingFace baixa `sentence-transformers/all-MiniLM-L6-v2` e o dataset `stanfordnlp/imdb`.

Testes unitários (opcional, após `make setup`):

```bash
python testes.py
```

## Uso

Script principal ([`src/experimento.py`](src/experimento.py)):

```bash
make experiment
python src/experimento.py --max-amostras 400 --max-tokens 256
python src/experimento.py --sanidade-gpt2   # opcional: checagem GPT-2
SEED=42 make experiment   # seed do split treino/teste (variavel de ambiente)
```

Grade experimental:

- 400 reviews balanceados (200 neg / 200 pos), split estratificado 70/30 (seed 42)
- 4 métodos × 3 limiares ε ∈ {0.90, 0.95, 0.99} × 3 orçamentos ρ ∈ {0.50, 0.25, 0.125} → **36 configurações** (22 execuções efetivas de operador)
- Saídas em `resultados/` (não versionadas — ver `.gitignore`):
  - `resultados.csv` / `resultados_imdb.csv`
  - `tradeoff_acerto_compressao.png` (facetas por ε)
  - `epsilon_k_cortes.png` (curva espectral e cortes k(ε))

Notebooks pedagógicos em [`research/`](research/): abrir no Jupyter com kernel **Python (svd-leverage)** (registrado por `make setup`).

## Notebooks

| Notebook | Conteúdo |
|----------|----------|
| [`research/extracao_embedding.ipynb`](research/extracao_embedding.ipynb) | Extração de $F$ etapa a etapa (`src/embeddings.py`: tokenização → MiniLM → filtro) |
| [`research/compressao_tokens.ipynb`](research/compressao_tokens.ipynb) | Poda de tokens etapa a etapa (`src/compressores.py`: baselines, SVD, leverage) |
| [`research/avaliacao_metricas.ipynb`](research/avaliacao_metricas.ipynb) | Métricas etapa a etapa (`src/metricas.py`: $C$, $E_k$, $R_k$, Ridge, acurácia) |
| [`research/experimento.ipynb`](research/experimento.ipynb) | Grade experimental etapa a etapa (`src/experimento.py`: orquestração, agregação, CSV) |
| [`research/trabalho_final.ipynb`](research/trabalho_final.ipynb) | Análise compilada: grade experimental e validação vs artigo |

## Resultados esperados

Acurácia downstream de referência (artigo, por ε):

**ε = 0,90**

| Método | ρ=0,50 | ρ=0,25 | ρ=0,125 |
|--------|--------|--------|---------|
| full | 0,750 | 0,750 | 0,750 |
| random | 0,725 | 0,775 | 0,700 |
| svd | 0,750 | 0,733 | 0,683 |
| svd_energia | 0,742 | 0,725 | 0,675 |

**ε = 0,95**

| Método | ρ=0,50 | ρ=0,25 | ρ=0,125 |
|--------|--------|--------|---------|
| full | 0,750 | 0,750 | 0,750 |
| random | 0,725 | 0,775 | 0,700 |
| svd | 0,783 | 0,742 | 0,725 |
| svd_energia | 0,742 | 0,733 | 0,675 |

**ε = 0,99**

| Método | ρ=0,50 | ρ=0,25 | ρ=0,125 |
|--------|--------|--------|---------|
| full | 0,750 | 0,750 | 0,750 |
| random | 0,725 | 0,775 | 0,700 |
| svd | 0,792 | 0,775 | 0,725 |
| svd_energia | 0,750 | 0,725 | 0,675 |

Referências auxiliares: $\bar{k}$ ≈ 34 / 50 / 92 e $E_k^{\mathrm{energia}}$ ≈ 0,903 / 0,951 / 0,990 para ε = 0,90 / 0,95 / 0,99; $\bar{T}$ ≈ 209 tokens/review; D = 384.

## Estrutura

| Arquivo | Função |
|---------|--------|
| `pyproject.toml` | Dependências e install editável (`pip install -e .`) |
| `Makefile` | Atalhos: `setup`, `experiment` |
| `src/constantes.py` | Parâmetros centralizados |
| `src/dados.py` | IMDb balanceado + split estratificado |
| `src/embeddings.py` | MiniLM token-level → matriz $F$ (sem modo sintético) |
| `src/compressores.py` | 4 operadores de poda |
| `src/metricas.py` | Acurácia, E_k, R_k, compressão |
| `src/experimento.py` | Grade experimental (CLI) |
| `src/visualizacao.py` | Figuras trade-off e cortes ε–k |
| `src/llm_sanidade.py` | Checagem exploratória GPT-2 (opcional) |
| `testes.py` | Testes unitários |
| `research/extracao_embedding.ipynb` | Notebook: extração de $F$ passo a passo |
| `research/compressao_tokens.ipynb` | Notebook: operadores de poda passo a passo |
| `research/avaliacao_metricas.ipynb` | Notebook: métricas e avaliação downstream passo a passo |
| `research/experimento.ipynb` | Notebook: orquestração da grade experimental passo a passo |
| `research/trabalho_final.ipynb` | Notebook: resultados e validação do artigo |

Artefatos CSV/PNG não são versionados — regenerar com `make experiment`. Detalhes metodológicos e validação automática: ver notebook.
