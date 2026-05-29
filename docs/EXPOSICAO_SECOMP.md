# Roteiro de Exposicao SECOMP

## Objetivo no estande

Fazer o visitante entender em menos de 60 segundos que o projeto usa PDI para transformar sinais faciais em indicadores educacionais, com uma demonstracao visual clara e tecnicamente defensavel.

## Setup fisico

- Notebook ligado na tomada.
- Webcam posicionada na altura dos olhos.
- Iluminacao frontal simples, sem contraluz forte.
- Poster A0 ao lado do notebook.
- Control Center aberto pelo `.\run.bat` no Windows ou `./run.sh` no Linux/macOS.
- Terminal aberto na pasta do projeto como fallback.
- Janela do app em tela cheia ou maximizada.

## Checagem 10 minutos antes

```powershell
.\run.bat
.\scripts\python.bat scripts/doctor.py --camera-check
.\scripts\python.bat -m edutech_vision --mode individual --showcase --no-sound --fullscreen --face-confidence 0.70
```

Linux/macOS:

```bash
./run.sh
./scripts/python.sh scripts/doctor.py --camera-check
./scripts/python.sh -m edutech_vision --mode individual --showcase --no-sound --fullscreen --face-confidence 0.70
```

Se FPS estiver baixo, reduzir resolucao:

```powershell
.\scripts\python.bat -m edutech_vision --mode individual --showcase --width 640 --height 480 --no-sound --face-confidence 0.70
```

Se a camera falhar, abrir fallback sintetico e avisar antes de mostrar:

```powershell
.\scripts\python.bat -m edutech_vision --mode demo --showcase --fullscreen
```

## Pitch leigo - 1 minuto

"Este projeto usa prior art de visao computacional: primeiro localiza rostos, inclusive menores e mais distantes, depois extrai pontos faciais em tempo real. No modo individual, ele mede fechamento dos olhos, abertura da boca e inclinacao da cabeca. No modo plateia, ele agrega quantas faces com pose valida estao voltadas para o palco em uma janela de 10 segundos."

## Demo recomendada - 3 minutos

1. Rodar Modo Individual.
2. Mostrar calibracao inicial de 5 segundos.
3. Piscar/fechar olhos por mais de 1 segundo para disparar fadiga.
4. Inclinar a cabeca para mostrar postura.
5. Virar o rosto para mostrar desatencao.
6. Alternar para Modo Plateia ou explicar com o painel de agregacao.
7. Abrir `results/benchmark/summary.html` para mostrar ganho medido de rostos distantes e a coluna de landmarks validos.
8. Abrir `results/session_report.pdf` para mostrar graficos, timeline e resumo executivo.

Atalho recomendado: usar os botoes `Modo Individual`, `Modo Plateia`, `Modo Demo` e a aba `Resultados` no Control Center.

## Explicacao tecnica - 5 minutos

1. YuNet localiza faces no perfil padrao e cria recortes ampliados.
2. MediaPipe Face Landmarker extrai landmarks faciais nos recortes.
3. EAR mede fechamento ocular por razao entre distancias verticais e horizontal do olho.
4. MAR mede abertura de boca.
5. `solvePnP` estima yaw/pitch/roll.
6. Media movel e estados sustentados reduzem ruido.
7. Modo Plateia separa faces detectadas de faces com pose valida e agrega engajamento por janela.

## Plano de contingencia

| Problema | Acao |
| --- | --- |
| Webcam falha | Mostrar tela de reconexao e trocar para webcam reserva |
| FPS baixo | Usar `--width 640 --height 480` e `--max-faces 6` |
| Iluminacao ruim | Reposicionar notebook ou usar lanterna/luminaria frontal |
| Detector nao abre | Rodar `scripts/doctor.py` e confirmar modelos MediaPipe/YuNet |
| Som incomoda | Usar `--no-sound` |
| Plateia distante | Usar detector `enhanced`; caixas sem pose continuam visiveis e nao distorcem o percentual |
| Webcam indisponivel | Usar `--mode demo --showcase` e apresentar como demo sintetica |

## Divisao sugerida da dupla

- Pessoa A: opera notebook, executa comandos e demonstra falhas controladas.
- Pessoa B: conversa com visitantes, aponta poster e responde metodologia/limitacoes.
- Na sabatina, alternar: quem nao operou explica codigo.
