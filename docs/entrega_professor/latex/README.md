# Fonte LaTeX do Artigo

Esta pasta contem a versao canonica do artigo no modelo SBC.

- `artigo_edutech_vision_grupo3.tex`: texto do artigo.
- `artigo_edutech_vision_grupo3.bib`: referencias no formato BibTeX usadas pelo estilo `sbc`.
- `sbc-template.sty` e `sbc.bst`: arquivos do template SBC.
- `pipeline_original.png`: figura usada no artigo.

Para recompilar localmente com Tectonic:

```powershell
tectonic -X compile artigo_edutech_vision_grupo3.tex
```

O PDF final entregue ao professor fica em `docs/entrega_professor/artigo_edutech_vision_grupo3.pdf`.
