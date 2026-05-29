# Checklist de Software para Apresentacao

Esta checklist cobre apenas o lado de software: ambiente, execucao, evidencias, logs, protocolos, demonstracao e contingencias tecnicas.

## 0. Preflight automatico

- [ ] Abrir o Control Center com `.\run.bat` no Windows ou `./run.sh` no Linux/macOS.
- [ ] Rodar o preflight automatizado antes de qualquer ensaio ou apresentacao.
- [ ] Ler `results/presentation_check_report.md`.
- [ ] Corrigir qualquer item `FAIL`.
- [ ] Tratar itens `WARN` antes da entrega final, principalmente resultados reais dos protocolos.
- [ ] Executar manualmente os itens marcados como `MANUAL`, pois dependem de webcam, ambiente fisico ou apresentacao oral.

Comando completo:

```powershell
.\run.bat
.\scripts\python.bat scripts/presentation_check.py
```

Linux/macOS:

```bash
./run.sh
./scripts/python.sh scripts/presentation_check.py
```

Comando completo incluindo teste de webcam:

```powershell
.\scripts\python.bat scripts/presentation_check.py --camera-check
```

Comando rapido, pulando lint/testes mais lentos:

```powershell
.\scripts\python.bat scripts/presentation_check.py --skip-slow
```

## 1. Estado do repositorio

- [ ] `git status --short` nao mostra mudancas inesperadas em arquivos rastreados.
- [ ] Existe commit recente com a versao que sera apresentada.
- [ ] Existe tag de marco, por exemplo `exhibition-readiness-0.1`, `n1-pre-avaliacao` ou `secomp-final`.
- [ ] `README.md` tem instrucoes de instalacao e execucao corretas.
- [ ] `docs/RUBRICA_MAX_NOTA.md` esta atualizado com evidencias reais.
- [ ] `docs/DEFESA_ORAL.md` esta disponivel para revisao rapida.
- [ ] `docs/EXPOSICAO_SECOMP.md` esta disponivel com roteiro de demo.
- [ ] Arquivos de midia bruta ou pesada nao estao versionados por acidente.
- [ ] `assets/models/face_landmarker.task` e `assets/models/face_detection_yunet_2023mar.onnx` existem localmente, mas continuam ignorados pelo Git.
- [ ] `assets/benchmarks/` existe localmente para evidencias e continua ignorado pelo Git.
- [ ] `results/` contem somente evidencias numericas e agregadas adequadas.

Comandos:

```powershell
git status --short
git log --oneline --decorate -5
git tag --list
```

## 2. Ambiente Python

- [ ] Ambiente `.venv` existe.
- [ ] Python do ambiente esta entre 3.10 e 3.11.
- [ ] Dependencias estao nas versoes pinadas em `requirements.txt`.
- [ ] Pacote local esta instalado em modo editavel.
- [ ] `.\scripts\python.bat -m edutech_vision --help` funciona no Windows; `./scripts/python.sh -m edutech_vision --help` funciona no Linux/macOS.
- [ ] `.\run.bat` abre o Control Center no Windows.
- [ ] `./run.sh` abre o Control Center no Linux/macOS, ou aponta claramente a dependencia grafica do sistema que falta.
- [ ] CLI mostra `demo` como modo sintetico de contingencia.
- [ ] `scripts/doctor.py` passa sem erros.
- [ ] Os modelos MediaPipe e YuNet do perfil `enhanced` foram baixados.
- [ ] O seletor de detector do Control Center esta em `enhanced` para a apresentacao.

Comandos:

```powershell
.\scripts\python.bat --version
.\scripts\python.bat scripts/doctor.py
.\scripts\python.bat scripts/presentation_check.py --camera-check
.\scripts\python.bat -m edutech_vision --help
```

Se o modelo estiver ausente:

```powershell
.\scripts\python.bat scripts/download_models.py
```

## 3. Validacao tecnica antes de apresentar

- [ ] `ruff` passa.
- [ ] `pytest` passa.
- [ ] `compileall` passa.
- [ ] Detector padrao `enhanced` (YuNet + MediaPipe em recortes) inicializa.
- [ ] Relatorio `results/benchmark/summary.html` mostra os gates do detector padrao.
- [ ] Matriz de confusao demo ou real gera saida sem erro.
- [ ] O app abre em Modo Individual.
- [ ] O app abre em Modo Plateia.
- [ ] O app abre em Modo Demo sintetico, sempre rotulado como simulacao.

Comandos:

```powershell
.\scripts\python.bat -m ruff check src tests scripts
.\scripts\python.bat -m pytest -q
.\scripts\python.bat -m compileall src tests scripts
.\scripts\python.bat -c "from edutech_vision.core.landmarks import FaceLandmarkDetector; d=FaceLandmarkDetector(max_faces=1, detector_profile='enhanced'); d.close(); print('detector_ok')"
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector enhanced --max-faces 24
```

## 4. Comandos principais de apresentacao

Modo Individual padrao:

```powershell
.\scripts\python.bat -m edutech_vision --mode individual --showcase --no-sound --fullscreen --face-confidence 0.70
```

Modo Individual com menor resolucao, caso FPS caia:

```powershell
.\scripts\python.bat -m edutech_vision --mode individual --showcase --width 640 --height 480 --no-sound --face-confidence 0.70
```

Modo Plateia padrao:

```powershell
.\scripts\python.bat -m edutech_vision --mode plateia --showcase --fullscreen
```

Modo Plateia para sala grande:

```powershell
.\scripts\python.bat -m edutech_vision --mode plateia --showcase --fullscreen --max-faces 24 --face-confidence 0.60 --audience-yaw-tolerance 30 --audience-pitch-tolerance 20
```

Modo Demo sintetico, apenas para fallback sem camera:

```powershell
.\scripts\python.bat -m edutech_vision --mode demo --showcase --fullscreen
```

Modo com video local de teste:

```powershell
.\scripts\python.bat -m edutech_vision --mode plateia --video assets/private/plateia_teste.mp4
```

Modo com video publico do corpus:

```powershell
.\scripts\python.bat -m edutech_vision --mode plateia --video assets/benchmarks/videos/classroom_students_ghana.webm --showcase
```

## 5. Fluxo da demo ao vivo

- [ ] Abrir terminal na raiz do projeto.
- [ ] Abrir Control Center.
- [ ] Rodar `scripts/doctor.py`.
- [ ] Rodar Modo Individual.
- [ ] Aguardar calibracao inicial de 5 segundos.
- [ ] Mostrar face detectada e landmarks/pontos principais.
- [ ] Mostrar EAR mudando ao fechar os olhos.
- [ ] Mostrar MAR mudando ao abrir a boca.
- [ ] Mostrar yaw/pitch mudando ao virar/inclinar a cabeca.
- [ ] Demonstrar alerta de fadiga sustentada sem depender de piscada curta.
- [ ] Demonstrar alerta de postura com inclinacao sustentada.
- [ ] Demonstrar alerta de desatencao virando o rosto por tempo suficiente.
- [ ] Fechar com `Q` ou `Esc`, sem matar processo pelo terminal.
- [ ] Rodar Modo Plateia.
- [ ] Mostrar contagem de faces, barras e percentual agregado.
- [ ] Mostrar timeline de eventos no painel lateral.
- [ ] Mostrar que o relatorio PDF/HTML nasce dos logs CSV.
- [ ] Abrir o resumo de benchmark e diferenciar `recall face` de `recall landmarks`.
- [ ] Se a webcam falhar, rodar Modo Demo e apresentar como demo sintetica.
- [ ] Abrir `results/session_report.pdf` para mostrar graficos, timeline e resumo executivo.

## 6. Evidencias que devem estar prontas para abrir

- [ ] `results/fps_log.csv`
- [ ] `results/lighting_evaluation.csv`
- [ ] `results/occlusion_recovery.csv`
- [ ] `results/confusion_metrics.csv`
- [ ] `results/confusion_matrix.png`
- [ ] `results/failover_log.csv`
- [ ] `results/protocol_summary.md`
- [ ] `results/presentation_check_report.md`
- [ ] `results/session_report.md`
- [ ] `results/session_report.html`
- [ ] `results/session_report.pdf`
- [ ] `results/benchmark/summary.html`
- [ ] `results/benchmark/false_negatives_contact_sheet.png`
- [ ] `results/benchmark/false_positives_contact_sheet.png`
- [ ] `results/report_charts/session_summary_dashboard.png`
- [ ] `results/report_charts/session_individual_timeline.png`
- [ ] `results/report_charts/session_audience_timeline.png`
- [ ] `results/report_charts/session_alerts.png`
- [ ] `docs/RELATORIO_TECNICO.md`
- [ ] `docs/entrega_professor/artigo_edutech_vision_grupo3.pdf`
- [ ] `docs/entrega_professor/artigo_edutech_vision_grupo3.docx`
- [ ] `docs/entrega_professor/poster_secomp_grupo3.pdf`
- [ ] `docs/entrega_professor/poster_secomp_grupo3.png`

Gerar resumo:

```powershell
.\scripts\python.bat scripts/summarize_results.py
.\scripts\python.bat scripts/generate_session_report.py
```

Gerar demonstracao sintetica do formato do relatorio:

```powershell
.\scripts\python.bat scripts/generate_session_report.py --demo-synthetic
```

Observacao: `results/demo_showcase/` e simulacao para demonstrar o formato. Para entrega, gere novamente com CSVs reais.

## 7. Protocolos ao vivo

### Robustez luminosa

- [ ] Executar baixa, media e alta iluminacao.
- [ ] Confirmar que nao houve crash.
- [ ] Confirmar que o CSV registrou os tres estagios.
- [ ] Explicar que a degradacao deve ser medida contra a condicao ideal.

```powershell
.\scripts\python.bat tests/test_lighting.py --mode individual
```

### Oclusao

- [ ] Registrar baseline.
- [ ] Bloquear a face por 3 segundos.
- [ ] Remover bloqueio.
- [ ] Confirmar redeteccao em ate 2 segundos.

```powershell
.\scripts\python.bat tests/test_occlusion.py --mode individual
```

### FPS

- [ ] Rodar por 60 segundos.
- [ ] Confirmar media >= 20 FPS.
- [ ] Reportar media, desvio-padrao, minimo e maximo.

```powershell
.\scripts\python.bat tests/test_fps.py --mode individual --duration 60
```

### Matriz de confusao

- [ ] Se a matriz de confusao presencial for usada, preencher `assets/samples/labels_real.csv`; caso contrario, usar `labels_demo.csv` como demonstracao controlada.
- [ ] Confirmar pelo menos 50 amostras.
- [ ] Confirmar acuracia >= 70%.
- [ ] Mostrar matriz completa.

```powershell
.\scripts\python.bat tests/test_confusion_matrix.py --labels assets/samples/labels_demo.csv
```

### Tolerancia a falhas

- [ ] Rodar script.
- [ ] Desconectar webcam.
- [ ] Confirmar mensagem amigavel.
- [ ] Reconectar webcam.
- [ ] Confirmar retomada sem crash.

```powershell
.\scripts\python.bat tests/test_failover.py --duration 75
```

## 8. Impacto visual e escopo tecnico

- [ ] Painel em tela cheia com video grande e lateral legivel.
- [ ] Modo Individual mostra bounding box, landmarks principais e eixos de pose.
- [ ] Barras de EAR, MAR, yaw e pitch respondem imediatamente aos movimentos.
- [ ] Alertas mudam de estado apenas quando a condicao e sustentada.
- [ ] Modo Plateia mostra contagem de faces, atentos e percentual de engajamento.
- [ ] Timeline registra eventos discretos durante a demonstracao.
- [ ] Relatorio PDF/HTML abre rapido e tem graficos claros.
- [ ] Explicar que o sistema gera indicadores visuais, nao diagnostico medico.

## 9. Performance e estabilidade

- [ ] Comecar com resolucao `960x540`.
- [ ] Se FPS ficar abaixo de 20, reduzir para `640x480`.
- [ ] No Modo Plateia, reduzir `--max-faces` se a maquina estiver lenta.
- [ ] Evitar deixar outros apps pesados abertos.
- [ ] Rodar `test_fps.py` no mesmo notebook da apresentacao.
- [ ] Verificar se o detector `enhanced` inicializa antes da banca chegar.
- [ ] Abrir o resumo do benchmark e saber explicar recall, precisao, F1 e FPS.
- [ ] Manter terminal com comandos recentes no historico.

Opcoes de degradacao controlada:

```powershell
.\scripts\python.bat -m edutech_vision --mode individual --width 640 --height 480 --no-sound
.\scripts\python.bat -m edutech_vision --mode plateia --width 640 --height 480 --max-faces 6
```

## 10. Fallbacks de software

| Falha | Resposta |
| --- | --- |
| Comando direto nao encontra o pacote | Rodar `.\run.bat`/`./run.sh` ou reinstalar o pacote local com o Python da `.venv` |
| Modelo ausente | Rodar `.\run.bat`/`./run.sh` ou `scripts/download_models.py` pelo Python da `.venv` |
| FPS baixo | Reduzir resolucao e `--max-faces` |
| Webcam nao abre | Rodar `doctor.py --camera-check`; trocar `--camera 1` |
| Som atrapalha | Usar `--no-sound` |
| Plateia com faces pequenas | Usar `enhanced`; caixas detectadas sem pose ficam separadas do percentual de engajamento |
| MediaPipe falha ao importar | Confirmar Python 3.11 e reinstall por `requirements.txt` |
| CSV nao gerou | Confirmar permissao de escrita em `results/` e rerodar protocolo |

## 11. Arquivos de codigo para explicar se sorteados

- [ ] `src/edutech_vision/core/metrics.py`: EAR e MAR.
- [ ] `src/edutech_vision/core/pose.py`: `solvePnP`, matriz de camera e modelo facial 3D.
- [ ] `src/edutech_vision/core/filters.py`: media movel e condicoes sustentadas.
- [ ] `src/edutech_vision/core/aggregation.py`: janela temporal de engajamento.
- [ ] `src/edutech_vision/core/detection.py`: YuNet/SCRFD, NMS e recortes para faces pequenas.
- [ ] `src/edutech_vision/core/landmarks.py`: MediaPipe no frame baseline ou nos recortes detectados.
- [ ] `src/edutech_vision/modes/individual.py`: baseline, limiares e alertas.
- [ ] `src/edutech_vision/modes/audience.py`: calibracao do palco e agregacao.
- [ ] `src/edutech_vision/utils/camera.py`: reconexao de webcam.
- [ ] `src/edutech_vision/protocols.py`: execucao dos cinco protocolos.
- [ ] `scripts/benchmark_vision.py`: comparacao reprodutivel com WIDER FACE e videos publicos.

## 12. Frases tecnicas que precisam estar alinhadas

- [ ] "EAR e MAR sao razoes geometricas, nao classificadores treinados."
- [ ] "O perfil padrao usa YuNet para detectar faces e MediaPipe nos recortes; isso melhora faces pequenas sem treinar um modelo proprio."
- [ ] "A pose de cabeca vem de `solvePnP`, usando pontos 2D da imagem e um modelo 3D generico."
- [ ] "A suavizacao temporal reduz falsos positivos de eventos pontuais."
- [ ] "Alertas so disparam quando a condicao e sustentada."
- [ ] "Modo Plateia converte varias faces em percentual de engajamento por janela."
- [ ] "Uma face detectada sem pose confiavel aparece no painel, mas nao distorce o percentual de engajamento."
- [ ] "Os logs sao evidencias numericas para reproducibilidade e artigo."
- [ ] "O sistema e uma ferramenta academica de PDI, nao diagnostico clinico."

## 13. Nao fazer durante a apresentacao

- [ ] Nao prometer deteccao perfeita de atencao.
- [ ] Nao dizer que identifica aluno desatento individualmente no Modo Plateia.
- [ ] Nao usar `labels_demo.csv` como resultado cientifico final.
- [ ] Nao encerrar processo de forma abrupta se a webcam falhar; mostrar a reconexao.
- [ ] Nao deixar a explicacao virar defesa de limitacoes; voltar para pipeline, resultados e demo.

## 14. Encerramento apos a apresentacao

- [ ] Fechar app com `Q` ou `Esc`.
- [ ] Conferir logs gerados em `results/`.
- [ ] Rodar `scripts/summarize_results.py` se houve nova coleta.
- [ ] Criar tag final se a versao apresentada mudou.

Comandos:

```powershell
.\scripts\python.bat scripts/summarize_results.py
git status --short --ignored
```
