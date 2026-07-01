# Checagem de sanidade com LLM pequeno (apoio qualitativo, nao metrica principal).
#
# Stub deterministico por padrao; opcionalmente usa gpt2 se torch/transformers
# estiverem instalados.

import numpy as np


def sanidade_stub(n_amostras=5):
    """Retorna score neutro 3.0/5.0 para cada amostra (modo offline)."""
    return [3.0] * n_amostras


def sanidade_gpt2(textos, n_amostras=5):
    """
    Verifica se o modelo gpt2 produz logits finitos para os textos truncados.
    Score 5.0 se logits finitos, 1.0 caso contrario.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return sanidade_stub(n_amostras)

    nome = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModelForCausalLM.from_pretrained(nome)
    modelo.eval()

    scores = []
    for texto in textos[:n_amostras]:
        ids = tokenizer(
            texto[:512], return_tensors="pt", truncation=True, max_length=64
        )
        with torch.no_grad():
            saida = modelo(**ids)
        logits = saida.logits.numpy()
        ok = np.isfinite(logits).all() and logits.shape[1] > 0
        scores.append(5.0 if ok else 1.0)
    return scores


def executar_sanidade(textos, usar_gpt2=False, n_amostras=5):
    if usar_gpt2:
        return sanidade_gpt2(textos, n_amostras=n_amostras)
    return sanidade_stub(n_amostras)
