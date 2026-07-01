# Compilacao do artigo LaTeX

O ambiente atual **nao possui** `latexmk` nem `pdflatex` instalados.

Para gerar o PDF localmente:

```bash
sudo apt install texlive-latex-base texlive-latex-extra latexmk
cd Trabalho_Final/template_trabalho
latexmk -pdf main.tex
```

Saida esperada: `main.pdf` (~10 paginas).

Figuras em `figures/` (copiadas de `codigo/resultados/` apos experimento IMDb).
