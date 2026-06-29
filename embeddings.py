# Extracao do embedding intermediario F in R^{T x D} a partir de texto.
#
# Modo padrao: embedding SINTETICO e deterministico (so numpy).
# Modo real (opcional): transformers + all-MiniLM-L6-v2 (requer pip install).

import numpy as np

_MODELO_CACHE = {}


def gerar_embedding_sintetico(n_tokens=64, dim=48, posto=6, nivel_ruido=0.05, semente=0):
    """
    F = A B^T + eta,  A in R^{T x p}, B in R^{D x p}.
    Entradas de A, B ~ N(0,1); eta ~ N(0, nivel_ruido^2).
    """
    rng = np.random.default_rng(semente)

    A = rng.standard_normal((n_tokens, posto))
    B = rng.standard_normal((dim, posto))
    F = A @ B.T
    F += nivel_ruido * rng.standard_normal((n_tokens, dim))
    return F.astype(np.float64)


def centro_classe(rotulo, dim):
    mu = np.zeros(dim)
    mu[rotulo % dim] = 3.0
    return mu


def gerar_amostra_rotulada(rotulo, n_classes, n_tokens=64, dim=48, posto=6, semente=0):
    rng = np.random.default_rng(1000 * rotulo + semente)
    mu_y = centro_classe(rotulo, dim)
    F = gerar_embedding_sintetico(
        n_tokens=n_tokens, dim=dim, posto=posto, semente=semente + rotulo
    )
    F = F + mu_y
    return F, mu_y


def gerar_conjunto_sintetico(n_amostras=60, n_classes=3, n_tokens=64, dim=48, semente=0):
    amostras, rotulos = [], []
    for i in range(n_amostras):
        rotulo = i % n_classes
        F, _ = gerar_amostra_rotulada(
            rotulo, n_classes, n_tokens=n_tokens, dim=dim, semente=semente + i
        )
        amostras.append(F)
        rotulos.append(rotulo)
    return amostras, np.array(rotulos)


def _carregar_modelo(nome="sentence-transformers/all-MiniLM-L6-v2"):
    if nome in _MODELO_CACHE:
        return _MODELO_CACHE[nome]
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Modelo real indisponivel. Instale: pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModel.from_pretrained(nome)
    modelo.eval()
    _MODELO_CACHE[nome] = (tokenizer, modelo)
    return tokenizer, modelo


def _remover_especiais_e_padding(tokenizer, entradas, hidden):
    """Remove [CLS], [SEP] e padding; mantem apenas tokens de conteudo."""
    input_ids = entradas["input_ids"][0].tolist()
    attention = entradas.get("attention_mask")
    if attention is not None:
        mask = attention[0].cpu().numpy().astype(bool)
    else:
        mask = np.ones(len(input_ids), dtype=bool)

    especiais = set(tokenizer.all_special_ids)
    indices = [
        i for i, tid in enumerate(input_ids)
        if mask[i] and tid not in especiais
    ]
    if not indices:
        return hidden[0:1]
    return hidden[indices]


def texto_para_embedding(texto, modelo=None, max_tokens=256):
    nome = modelo or "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer, modelo_pt = _carregar_modelo(nome)

    import torch  # type: ignore

    entradas = tokenizer(
        texto, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    with torch.no_grad():
        saida = modelo_pt(**entradas)
    hidden = saida.last_hidden_state[0].cpu().numpy().astype(np.float64)
    F = _remover_especiais_e_padding(tokenizer, entradas, hidden)
    if F.shape[0] == 0:
        F = hidden[0:1]
    return F
