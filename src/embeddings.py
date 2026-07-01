"""
embeddings.py

Extrai matriz de embeddings token-level F ∈ R^{T x D} a partir de texto.

Pipeline: tokenização → MiniLM (last_hidden_state) → remoção de
[CLS]/[SEP]/padding → F.

Função principal do experimento: texto_para_embedding(...)
"""

from __future__ import annotations

from typing import Any

import numpy as np


MODELO_PADRAO = "sentence-transformers/all-MiniLM-L6-v2"

# Cache para não recarregar tokenizer/modelo a cada chamada.
_MODELO_CACHE: dict[str, tuple[Any, Any]] = {}


def carregar_modelo(modelo_nome: str = MODELO_PADRAO) -> tuple[Any, Any]:
    """
    Carrega tokenizer e modelo.

    Usa AutoModel porque queremos last_hidden_state,
    isto é, embeddings por token, não um vetor único da sentença.
    """
    if modelo_nome in _MODELO_CACHE:
        return _MODELO_CACHE[modelo_nome]

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Instale as dependências para o modo real: "
            "pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModel.from_pretrained(modelo_nome)
    modelo.eval()

    _MODELO_CACHE[modelo_nome] = (tokenizer, modelo)
    return tokenizer, modelo


def tokenizar_texto(
    texto: str,
    tokenizer: Any,
    max_tokens: int = 256,
) -> Any:
    """
    Tokeniza texto para PyTorch.

    max_tokens limita tokens do modelo, não palavras.
    truncation=True evita ultrapassar o tamanho máximo.
    """
    return tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        padding=False,
    )


def obter_tokens(tokenizer: Any, entradas: Any) -> list[str]:
    """
    Converte input_ids de volta para tokens legíveis.
    Útil para inspeção no notebook.
    """
    input_ids = entradas["input_ids"][0].tolist()
    return tokenizer.convert_ids_to_tokens(input_ids)


def extrair_hidden_states(
    modelo: Any,
    entradas: Any,
    dtype=np.float32,
) -> np.ndarray:
    """
    Executa o Transformer e retorna hidden ∈ R^{L x D}.

    L ainda inclui tokens especiais como [CLS] e [SEP].
    """
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Instale PyTorch: pip install torch") from exc

    with torch.no_grad():
        saida = modelo(**entradas)

    hidden = saida.last_hidden_state[0]
    return hidden.cpu().numpy().astype(dtype)


def indices_tokens_conteudo(
    tokenizer: Any,
    entradas: Any,
) -> list[int]:
    """
    Retorna posições que não são padding nem tokens especiais.
    """
    input_ids = entradas["input_ids"][0].tolist()

    attention_mask = entradas.get("attention_mask")
    if attention_mask is None:
        mask = np.ones(len(input_ids), dtype=bool)
    else:
        mask = attention_mask[0].cpu().numpy().astype(bool)

    especiais = set(tokenizer.all_special_ids)

    return [
        i
        for i, token_id in enumerate(input_ids)
        if mask[i] and token_id not in especiais
    ]


def selecionar_tokens_conteudo(
    hidden: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """
    Filtra hidden e retorna F ∈ R^{T x D}, apenas com tokens de conteúdo.
    """
    if not indices:
        return hidden[0:1]

    return hidden[indices]


def texto_para_embedding(
    texto: str,
    modelo_nome: str = MODELO_PADRAO,
    max_tokens: int = 256,
    dtype=np.float32,
) -> np.ndarray:
    """
    Função principal usada no programa final.

    texto
      -> tokenização
      -> Transformer
      -> hidden states
      -> remoção de especiais/padding
      -> F ∈ R^{T x D}
    """
    tokenizer, modelo = carregar_modelo(modelo_nome)

    entradas = tokenizar_texto(
        texto=texto,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )

    hidden = extrair_hidden_states(
        modelo=modelo,
        entradas=entradas,
        dtype=dtype,
    )

    indices = indices_tokens_conteudo(
        tokenizer=tokenizer,
        entradas=entradas,
    )

    F = selecionar_tokens_conteudo(
        hidden=hidden,
        indices=indices,
    )

    return F
