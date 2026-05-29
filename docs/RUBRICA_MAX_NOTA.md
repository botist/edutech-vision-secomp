# Rubrica e Evidencias para Maximizar Nota

Este documento transforma a rubrica do enunciado em evidencias concretas para mostrar ao professor e ao publico.

## N1 - Engenharia / Artefatos (4,0)

| Subitem | O que mostrar | Evidencia no repositorio | Risco | Acao obrigatoria |
| --- | --- | --- | --- | --- |
| Governanca Git | Historico com commits claros e tags | `git log --oneline`, `git tag` | Commit unico reduz nota | Criar commits por marco e tag `n1-pre-avaliacao` |
| Qualidade do codigo | Modulos pequenos e testados | `src/edutech_vision/`, `tests/unit/` | Nao saber explicar codigo | Ensaiar explicacao de cada modulo |
| Implementacao PDI | YuNet + recortes, landmarks, EAR, MAR, `solvePnP`, suavizacao e estados | `core/detection.py`, `core/metrics.py`, `core/pose.py`, `core/filters.py` | Professor sortear trecho | Abrir os arquivos e explicar formulas |
| Reprodutibilidade | Instalacao por um comando, modelos baixaveis, doctor e benchmark | `run.bat`, `run.sh`, `scripts/doctor.py`, `scripts/benchmark_vision.py` | Falha de instalacao | Rodar `doctor.py` e benchmark smoke antes da avaliacao |
| Robustez de deteccao | Comparacao baseline versus `enhanced` em imagens e videos publicos | `results/benchmark/summary.html` | Face pequena nao ser encontrada ao vivo | Mostrar ganhos e exemplos de falhas do benchmark |

## N1 - Testes ao Vivo (4,0)

| Protocolo | Como ganhar o ponto | Arquivo esperado | Comando |
| --- | --- | --- | --- |
| Robustez luminosa | Sem crash em baixa/media/alta; registrar deteccao | `results/lighting_evaluation.csv` | `python tests/test_lighting.py --mode individual` |
| Oclusao | Bloquear face 3 s e recuperar em ate 2 s | `results/occlusion_recovery.csv` | `python tests/test_occlusion.py --mode individual` |
| FPS | Media >= 20 FPS em 60 s | `results/fps_log.csv` | `python tests/test_fps.py --mode individual --duration 60` |
| Matriz de confusao | demonstracao controlada ou amostras reais, acuracia >= 70% | `results/confusion_metrics.csv` | `python tests/test_confusion_matrix.py --labels assets/samples/labels_demo.csv` |
| Failover | Desconectar/reconectar webcam sem crash | `results/failover_log.csv` | `python tests/test_failover.py --duration 75` |
| Deteccao publica | Recall/precisao/FPS contra baseline | `results/benchmark/summary.html` | `python scripts/benchmark_vision.py --suite smoke --mode plateia --detector enhanced --max-faces 24` |

Depois dos cinco testes:

```powershell
.\scripts\python.bat scripts/summarize_results.py
```

## N1 - Artigo Parte I (2,0)

| Subitem | Evidencia | Acao |
| --- | --- | --- |
| Template SBC | `docs/entrega_professor/artigo_edutech_vision_grupo3.pdf` e `.docx` | Conferir arquivos finais sem editar o conteudo |
| Metodologia | Secao de pipeline, limiares e protocolos | Substituir texto generico por detalhes da coleta real |
| Escrita e referencias | `referencias.bib` | Citar OpenCV/YuNet, MediaPipe, WIDER FACE, Gonzalez/Woods e Szeliski |

## N2 - Exposicao SECOMP (5,0)

| Subitem | Como maximizar | Evidencia |
| --- | --- | --- |
| Robustez in loco | Rodar `doctor.py`, baixar modelo antes, fonte de energia, webcam reserva | `scripts/doctor.py` |
| Sabatina tecnica | Qualquer integrante explica pipeline em 5 min | `docs/DEFESA_ORAL.md` |
| UX / Atendimento | Mostrar modo individual em 1 min e modo plateia com painel grande, barras e timeline | `docs/EXPOSICAO_SECOMP.md` |
| Inovacao | Dois modos reais, perfil YuNet para faces distantes, benchmark publico, painel `--showcase`, timeline, failover e relatorios | App + scripts |

## N2 - Poster (3,0)

| Subitem | Como maximizar | Evidencia |
| --- | --- | --- |
| Conteudo tecnico | Fluxograma PDI e resultados quantitativos visiveis | `docs/entrega_professor/poster_secomp_grupo3.pdf` e `.png` |
| Hierarquia | Leitura a 1,5 m, poucos blocos densos | Revisao impressa em A4 antes do A0 |
| Qualidade grafica | Paleta consistente e sem erros | Revisao por outra pessoa antes de imprimir |

## N2 - Artigo Parte II (2,0)

| Subitem | Como maximizar |
| --- | --- |
| Resultados | Usar tabelas, PDF e graficos dos cinco protocolos, nao apenas relato textual |
| Limitacoes | Explicar que os indicadores sao aproximacoes visuais, nao diagnostico |
| Defesa | Saber justificar limiares, calibracao, `solvePnP`, limitacoes e falhas |

## Ordem de Prioridade

1. Coletar resultados reais dos cinco protocolos.
2. Acrescentar os resultados presenciais ao artigo e poster, que ja incluem o benchmark publico.
3. Ensaiar defesa oral com trecho de codigo sorteado.
4. Rodar `doctor.py` e `test_fps.py` no local da apresentacao.
5. Criar tag Git antes da avaliacao.
