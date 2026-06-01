# EduTech Vision

Sistema de visao computacional em tempo real para demonstracoes educacionais.

Projeto final de Processamento Digital de Imagens - UNEMAT, 2026/1.  
Grupo 3: Fernando Cobianchi e Samuel Bernini.

## O Que Ele Faz

O EduTech Vision usa webcam, OpenCV e MediaPipe para transformar sinais visuais em indicadores educacionais ao vivo. O perfil padrao aplica YuNet antes dos landmarks para recuperar faces menores e mais distantes.

- **Modo Individual:** monitora uma pessoa por webcam e estima fadiga, bocejo, postura ruim e desatencao sustentada.
- **Modo Plateia:** estima o percentual de faces voltadas para o palco/professor em janelas de 10 segundos.
- **Modo Demo:** roda uma simulacao visual sem camera, util para apresentacao quando a webcam nao estiver disponivel.
- **Relatorio:** gera Markdown, HTML, PDF e graficos a partir dos CSVs da sessao.

O foco da interface e mostrar o pipeline funcionando: video anotado, landmarks, EAR/MAR, pose de cabeca, barras, timeline de eventos, FPS e relatorio grafico.

## Instalar E Abrir Em 1 Comando

Requisitos:

- Windows 10/11, Linux ou macOS 64-bit.
- PowerShell no Windows; Bash no Linux/macOS.
- Internet no primeiro uso.
- Git para clonar o repositorio.
- Webcam para os modos reais.

Windows:

```powershell
git clone https://github.com/botist/edutech-vision-secomp.git
cd edutech-vision-secomp
.\run.bat
```

Linux/macOS:

```bash
git clone https://github.com/botist/edutech-vision-secomp.git
cd edutech-vision-secomp
./run.sh
```

Esse unico comando prepara tudo e abre o Control Center. No Windows, se a maquina nao tiver Python 3.11, o projeto baixa `uv`, baixa um Python 3.11 local dentro de `.tools/`, cria `.venv`, instala dependencias e baixa os modelos, sem mexer no Python global e sem Visual Studio Build Tools. No Linux/macOS, o script usa Python 3.11 existente ou baixa `uv` localmente para criar um ambiente isolado.

Observacao para Linux/macOS: o setup instala as dependencias Python, mas sistemas desktop muito enxutos podem nao vir com Tkinter/OpenGL. Se o Control Center nao abrir, o proprio `./run.sh` mostra o pacote de sistema que falta; o CLI continua instalado.

O Control Center permite:

- rodar Modo Individual, Plateia e Demo;
- ajustar camera, resolucao, tela cheia, tolerancias e numero de faces;
- escolher detector `enhanced`, baseline `mediapipe` ou perfil experimental `research`;
- baixar modelo, reparar setup e rodar diagnostico;
- gerar relatorios e abrir arquivos de apresentacao;
- ver o log dos comandos sem precisar lembrar tudo de cabeca.

Configuracao padrao da demonstracao: detector `enhanced`; Modo Individual com confianca `0.70`; Modo Plateia com confianca `0.60`, ate `24` faces, tolerancia de yaw `30 deg` e pitch `20 deg`.

Depois da primeira execucao, `.\run.bat` ou `./run.sh` abre bem mais rapido porque `.venv`, dependencias e modelo ja ficam cacheados localmente.

## Rodar A Demo

Modo individual, em tela cheia:

```powershell
.\scripts\run_individual.ps1
```

Modo plateia, em tela cheia:

```powershell
.\scripts\run_plateia.ps1
```

Modo demo sintetico, sem webcam:

```powershell
.\scripts\run_demo.ps1
```

No Linux/macOS, use os equivalentes:

```bash
./scripts/run_individual.sh
./scripts/run_plateia.sh
./scripts/run_demo.sh
```

Sair da janela: pressione `Q` ou `Esc`.

## Comandos Diretos

Se preferir rodar diretamente:

```powershell
.\scripts\python.bat -m edutech_vision --mode individual --showcase --no-sound --fullscreen --face-confidence 0.70
.\scripts\python.bat -m edutech_vision --mode plateia --showcase --fullscreen --max-faces 24 --face-confidence 0.60 --audience-yaw-tolerance 30 --audience-pitch-tolerance 20
.\scripts\python.bat -m edutech_vision --mode demo --showcase --fullscreen
```

No Linux/macOS, troque `.\scripts\python.bat` por `./scripts/python.sh`.

Gerar relatorio da sessao:

```powershell
.\scripts\python.bat scripts\generate_session_report.py
```

Gerar relatorio de exemplo com dados sinteticos:

```powershell
.\scripts\python.bat scripts\generate_session_report.py --demo-synthetic
```

## Pipeline Tecnico

```text
Webcam ou video
  -> YuNet (padrao) ou SCRFD (pesquisa) detecta faces
    -> crop ampliado da face
  -> MediaPipe Face Landmarker
    -> landmarks faciais
    -> EAR e MAR
    -> pose de cabeca com solvePnP
    -> suavizacao temporal
    -> alertas sustentados ou agregacao de plateia
  -> painel OpenCV
  -> logs CSV
  -> relatorio HTML/PDF com graficos
```

Componentes principais:

- `src/edutech_vision/core/metrics.py`: EAR e MAR.
- `src/edutech_vision/core/detection.py`: YuNet/SCRFD, NMS e recortes de face.
- `src/edutech_vision/core/pose.py`: estimativa de yaw, pitch e roll com `cv2.solvePnP`.
- `src/edutech_vision/core/filters.py`: media movel e estados sustentados.
- `src/edutech_vision/core/aggregation.py`: janela temporal do modo plateia.
- `src/edutech_vision/modes/individual.py`: logica do modo individual.
- `src/edutech_vision/modes/audience.py`: logica do modo plateia.
- `src/edutech_vision/modes/demo.py`: simulacao visual.
- `src/edutech_vision/ui/renderer.py`: dashboard OpenCV.

## Arquivos Para Entrega

Os arquivos prontos para o professor ficam em `docs/entrega_professor/`:

- artigo em PDF e DOCX;
- poster em PDF e PNG.

## Testes E Preflight

Rodar verificacao completa:

```powershell
.\scripts\python.bat scripts\presentation_check.py --skip-slow
```

Rodar verificacao com webcam:

```powershell
.\scripts\python.bat scripts\presentation_check.py --camera-check --skip-slow
```

Instalar ferramentas de desenvolvimento:

```powershell
.\scripts\python.bat -m pip install -r requirements-dev.txt
```

Rodar testes unitarios e lint:

```powershell
.\scripts\python.bat -m pytest -q
.\scripts\python.bat -m ruff check src tests scripts
.\scripts\python.bat -m pip check
```

## Benchmark Com Imagens E Videos Reais

Baixar localmente WIDER FACE, tres videos livres de sala de aula e o perfil SCRFD:

```powershell
.\scripts\python.bat scripts\download_benchmarks.py
```

Comparar baseline e detector padrao com a configuracao atual:

```powershell
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector mediapipe --max-faces 24
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector enhanced --max-faces 24
```

O relatorio fica em `results/benchmark/summary.md`, com recall, precisao, F1, FPS em 960x540 e imagens de falhas. Um resumo de entrega esta em `docs/entrega_professor/RESULTADOS_BENCHMARK.md`. O perfil `research` usa pesos SCRFD/InsightFace para uso academico non-commercial; o perfil distribuido por padrao e `enhanced` com YuNet.

Auditoria manual de contagem em videos completos:

```powershell
.\scripts\python.bat scripts\manual_face_count_audit.py --detector enhanced --confidence 0.60 --run-label enhanced_c60_m24 --max-faces 24 --count-file results/manual_face_audit_hq/manual_counts.csv --videos-dir assets/benchmarks/hq_videos --output-dir results/manual_face_audit_hq --manifest-file assets/benchmarks/hq_videos/manifest.json
```

O material bruto sai em `results/manual_face_audit_hq/`: frames contados manualmente, prints comparativos, logs por frame em tempo real e `profile_summary.md`. Esses artefatos nao entram no GitHub; o resumo de entrega fica em `docs/entrega_professor/RESULTADOS_BENCHMARK.md`. O melhor perfil medido foi `enhanced_c60_m24`, que tambem e o padrao distribuido.

Auditoria HQ com videos 1080p e execucao em video inteiro:

```powershell
.\scripts\python.bat scripts\download_hq_audit_videos.py
.\scripts\python.bat scripts\manual_face_count_audit.py --detector enhanced --confidence 0.60 --run-label enhanced_c60_m24 --max-faces 24 --count-file results/manual_face_audit_hq/manual_counts.csv --videos-dir assets/benchmarks/hq_videos --output-dir results/manual_face_audit_hq --manifest-file assets/benchmarks/hq_videos/manifest.json
```

O relatorio HQ bruto fica em `results/manual_face_audit_hq/`. No sweep 1080p, `enhanced_c60_m24` foi o melhor perfil: erro medio 0.636 rosto, 10/11 casos com erro de no maximo 1 rosto e resposta correta aparecendo na janela temporal em 10/11 frames.

Benchmark smoke medido no notebook de desenvolvimento:

| Modo | Baseline recall face | Enhanced recall face | Enhanced recall landmarks | Precisao enhanced | FPS enhanced |
| --- | ---: | ---: | ---: | ---: | ---: |
| Individual | 0.193 | 0.592 | 0.294 | 0.894 | 29.0 |
| Plateia | 0.011 | 0.443 | 0.089 | 0.851 | 74.5 |

`Recall face` mede a localizacao visual da caixa. `Recall landmarks` mede as deteccoes que tambem sustentam EAR/MAR/pose; rostos distantes detectados sem landmarks aparecem no painel, mas nao entram no percentual de engajamento.

## Protocolos Experimentais

Os scripts abaixo geram resultados em `results/`:

```powershell
.\scripts\python.bat tests\test_lighting.py --mode individual
.\scripts\python.bat tests\test_occlusion.py --mode individual
.\scripts\python.bat tests\test_fps.py --mode individual --duration 60
.\scripts\python.bat tests\test_confusion_matrix.py --labels assets\samples\labels_demo.csv
.\scripts\python.bat tests\test_failover.py --duration 75
.\scripts\python.bat scripts\summarize_results.py
```

Os protocolos presenciais geram evidencias locais em `results/`. A matriz de confusao pode usar `assets/samples/labels_demo.csv` para demonstracao ou um CSV real rotulado no mesmo formato.

## Estrutura

```text
assets/      modelos locais e amostras rotuladas
docs/        roteiro, defesa oral, artigo e poster
results/     saidas geradas localmente
scripts/     setup, diagnostico, relatorios e atalhos
src/         pacote edutech_vision
tests/       protocolos e testes unitarios
```

## Escopo

O EduTech Vision e uma ferramenta academica de PDI. Os indicadores sao aproximacoes visuais baseadas em geometria facial, pose de cabeca e comportamento temporal. A qualidade depende de iluminacao, distancia da camera, FPS, resolucao e calibracao inicial.

## Referencias

- Gonzalez, R. C.; Woods, R. E. Processamento Digital de Imagens. Pearson, 2018.
- Szeliski, R. Computer Vision: Algorithms and Applications. Springer, 2022.
- OpenCV Documentation: https://docs.opencv.org
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
