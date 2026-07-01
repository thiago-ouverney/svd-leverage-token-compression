TRABALHO FINAL ALC � INSTRUCOES OVERLEAF
=========================================

1. Faca upload do ZIP neste projeto (Upload Project).

2. IMPORTANTE � Main document:
   Menu (canto superior esquerdo) > Main document > selecione "main.tex"
   NAO compile "preamble.tex" (ele nao tem \documentclass sozinho).

3. Compilador: pdfLaTeX (padrao do Overleaf).

4. Bibliografia: o Overleaf roda BibTeX automaticamente ao ver \bibliography{refs}.
   Se as referencias aparecerem como "?", recompile mais uma vez.

5. Antes de entregar, edite em main.tex:
   - Nome e email do autor
   - Link do repositorio na secao Codigo

Estrutura do projeto:
  main.tex          <- arquivo principal
  preamble.tex      <- pacotes (nao compilar direto)
  refs.bib          <- bibliografia
  sbc-template.sty  <- estilo SBC/UFF
  sections/         <- secoes do artigo
  figures/          <- graficos PNG
