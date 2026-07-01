# SVD + leverage scores — compressão de tokens

Trabalho Final de **Álgebra Linear Computacional** (UFF 2026.1): compressão token-level de embeddings via SVD truncada e leverage scores, com baselines e avaliação downstream em IMDb.

**Repositório:** [github.com/thiago-ouverney/svd-leverage-token-compression](https://github.com/thiago-ouverney/svd-leverage-token-compression)

## Pipeline

```
review IMDb → F (T×D) → poda (6 operadores) → Ridge → métricas
```

Operadores de poda:

- `full` — sem poda (teto de referência)
- `random` — tokens aleatórios
- `norm` — maior norma L2 por linha
- `first` — primeiros T′ tokens
- `svd` — leverage scores ℓᵢ/k (SVD truncada a 95% de energia)
- `svd_energia` — leverage ponderado por Σ

## Instalação

Com Makefile (recomendado):

```bash
make install          # deps mínimas
make test             # testes unitários

make install-experiment   # deps completas
make experiment           # grade IMDb (~229 s CPU)
```

Equivalente manual:

```bash
pip install -r requirements.txt && python testes.py
pip install datasets transformers torch && python experimento.py
```

Na primeira execução do experimento, o HuggingFace baixa `sentence-transformers/all-MiniLM-L6-v2` e o dataset `stanfordnlp/imdb`. Jupyter não está em `requirements.txt` — necessário só para o notebook.

## Uso

Script principal (`experimento.py`):

```bash
python experimento.py --max-amostras 400 --max-tokens 256
python experimento.py --sanidade-gpt2   # opcional: checagem GPT-2
SEED=42 SEMENTES=0,1,2 python experimento.py   # seeds via variaveis de ambiente
```

Grade experimental:

- 400 reviews balanceados (200 neg / 200 pos), split estratificado 70/30 (seed 42)
- 6 métodos × 4 orçamentos ρ ∈ {0.75, 0.5, 0.25, 0.125} × 3 seeds → **72 runs**
- Saídas em `resultados/` (não versionadas — ver `.gitignore`):
  - `resultados.csv` / `resultados_imdb.csv`
  - `tradeoff_acerto_compressao.png`

Notebooks pedagógicos: [`extracao_embedding.ipynb`](extracao_embedding.ipynb) (extração de $F$), [`compressao_tokens.ipynb`](compressao_tokens.ipynb) (operadores de poda), [`avaliacao_metricas.ipynb`](avaliacao_metricas.ipynb) (métricas e downstream) e [`trabalho_final.ipynb`](trabalho_final.ipynb) (grade e resultados). Com `FORCAR_REEXECUCAO = False`, reutiliza CSV em cache.

## Notebooks

| Notebook | Conteúdo |
|----------|----------|
| [`extracao_embedding.ipynb`](extracao_embedding.ipynb) | Extração de $F$ etapa a etapa (`embeddings.py`: tokenização → MiniLM → filtro) |
| [`compressao_tokens.ipynb`](compressao_tokens.ipynb) | Poda de tokens etapa a etapa (`compressores.py`: baselines, SVD, leverage) |
| [`avaliacao_metricas.ipynb`](avaliacao_metricas.ipynb) | Métricas etapa a etapa (`metricas.py`: $C$, $E_k$, $R_k$, Ridge, acurácia) |
| [`trabalho_final.ipynb`](trabalho_final.ipynb) | Análise compilada: grade experimental e validação vs artigo |

## Resultados esperados

Acurácia downstream de referência (artigo):

| Método | ρ=0.75 | ρ=0.5 | ρ=0.25 | ρ=0.125 |
|--------|--------|-------|--------|---------|
| full | 0.717 | 0.717 | 0.717 | 0.717 |
| random | 0.714 | 0.708 | 0.706 | 0.711 |
| norm | 0.725 | 0.700 | 0.667 | 0.650 |
| first | 0.742 | 0.758 | 0.725 | 0.683 |
| svd | 0.733 | 0.717 | 0.683 | 0.675 |
| svd_energia | 0.725 | 0.683 | 0.675 | 0.650 |

Referências auxiliares: E_k^energia ≈ 0,951; T̄ ≈ 209 tokens/review; D = 384.

## Estrutura

| Arquivo | Função |
|---------|--------|
| `Makefile` | Atalhos: `install`, `test`, `experiment` |
| `dados.py` | IMDb balanceado + split estratificado |
| `embeddings.py` | MiniLM token-level → matriz $F$ (sem modo sintético) |
| `compressores.py` | 6 operadores de poda |
| `metricas.py` | Acurácia, E_k, R_k, compressão |
| `experimento.py` | Grade experimental (CLI) |
| `visualizacao.py` | Figura trade-off |
| `llm_sanidade.py` | Checagem exploratória GPT-2 (opcional) |
| `testes.py` | Testes unitários |
| `extracao_embedding.ipynb` | Notebook: extração de $F$ passo a passo |
| `compressao_tokens.ipynb` | Notebook: operadores de poda passo a passo |
| `avaliacao_metricas.ipynb` | Notebook: métricas e avaliação downstream passo a passo |
| `trabalho_final.ipynb` | Notebook: resultados e validação do artigo |

Artefatos CSV/PNG não são versionados — regenerar com `make experiment` ou `python experimento.py`. Detalhes metodológicos e validação automática: ver notebook.
