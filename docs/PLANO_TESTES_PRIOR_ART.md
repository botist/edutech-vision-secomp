# Plano De Testes E Melhoria Com Prior Art

## Resumo

Vamos usar modelos prontos de alta qualidade para atacar o gargalo real do projeto: deteccao de faces pequenas, distantes, parcialmente ocluidas ou em plateia.

O app continua usando MediaPipe para landmarks, EAR/MAR e pose com `solvePnP`. A mudanca principal sera antes disso: melhorar a deteccao inicial de face e alimentar o MediaPipe com regioes de interesse melhores.

Como o projeto e academico, podemos usar modelos gratuitos de pesquisa, desde que a documentacao deixe claro quando um modelo tem licenca de uso non-commercial/research.

## Objetivo

Criar uma versao mensuravelmente mais robusta do Modo Individual e do Modo Plateia, validada por benchmarks offline com imagens e videos conhecidos.

O foco nao e "parecer melhor" no olho. O foco e medir:

- onde o detector atual falha;
- quais cenarios melhoram com prior art;
- se houve regressao em cenarios que ja funcionavam;
- se o FPS continua aceitavel para apresentacao.

## Prior Art Que Vamos Usar

### MediaPipe Face Landmarker

Uso atual do projeto. Continuara responsavel por:

- landmarks faciais;
- EAR;
- MAR;
- pose por `solvePnP`;
- visualizacao dos pontos principais.

Limite conhecido: sozinho, ele tende a falhar quando a face esta pequena/distante no frame ou quando a deteccao inicial nao encontra uma face confiavel.

### OpenCV YuNet

Detector de face leve, rapido e gratuito, disponivel no OpenCV Zoo.

Uso planejado:

- detector auxiliar padrao no perfil `enhanced`;
- bom equilibrio entre instalacao simples, performance e recall;
- roda via `opencv-contrib-python`, que ja esta no projeto.

Modelo:

- `face_detection_yunet_2023mar.onnx`
- destino local: `assets/models/face_detection_yunet_2023mar.onnx`

### InsightFace SCRFD

Detector de face mais forte, especialmente interessante para faces pequenas e cenarios dificeis.

Uso planejado:

- perfil experimental `research`;
- roda via `onnxruntime`;
- deve ter aviso claro de licenca non-commercial/research.

Modelo inicial recomendado:

- `SCRFD_2.5G_KPS` ou variante equivalente em ONNX;
- usar como detector de alta precisao para comparacao com YuNet e MediaPipe.

## Perfis De Detector

### `mediapipe`

Baseline atual.

Uso:

- comparar contra o comportamento existente;
- provar numericamente que as melhorias fazem diferenca.

### `enhanced`

Novo padrao do app.

Pipeline:

1. Detectar faces com YuNet.
2. Aplicar NMS e filtro de duplicatas.
3. Criar crops com padding ao redor das faces.
4. Rodar MediaPipe Face Landmarker em cada crop.
5. Remapear landmarks para coordenadas do frame original.
6. No Modo Individual, escolher a face maior e mais central.
7. No Modo Plateia, processar ate `max_faces`.

### `research`

Modo experimental de maior recall.

Pipeline:

1. Detectar faces com SCRFD.
2. Aplicar NMS e filtro de duplicatas.
3. Criar crops com padding.
4. Rodar MediaPipe no crop.
5. Remapear landmarks.
6. Usar o mesmo fluxo de metricas dos outros modos.

Esse modo nao deve ser o unico caminho do app. Ele serve para obter a maior taxa de acerto possivel e comparar contra o perfil leve.

## Mudancas No App

### CLI

Adicionar:

```powershell
--detector mediapipe|enhanced|research
--face-confidence 0.70  # Individual
--face-confidence 0.60  # Plateia
```

Default:

```text
--detector enhanced
Individual: --face-confidence 0.70
Plateia: --face-confidence 0.60 --max-faces 24
```

### Control Center

Adicionar seletor simples:

```text
Detector: Enhanced / MediaPipe / Research
```

Default:

```text
Enhanced
```

### Setup

`run.bat` no Windows e `run.sh` no Linux/macOS continuam sendo os comandos unicos de entrada.

Ele deve:

- manter instalacao simples no Windows 10/11;
- manter instalacao simples em Linux/macOS desktop 64-bit;
- instalar `onnxruntime` se o perfil `research` for suportado diretamente;
- baixar YuNet sempre;
- baixar SCRFD quando necessario ou por comando de preparacao;
- manter modelos grandes ignorados pelo Git.

## Benchmark Offline

### Downloader

Criar:

```text
scripts/download_benchmarks.py
```

Responsabilidades:

- baixar WIDER FACE validacao e anotacoes;
- baixar videos livres do Wikimedia Commons;
- baixar YuNet;
- baixar SCRFD;
- gerar manifesto com URL, licenca, hash e tamanho;
- salvar tudo em `assets/benchmarks/` e `assets/models/`.

Os arquivos baixados nao devem ser commitados.

### Runner

Criar:

```text
scripts/benchmark_vision.py
```

Comandos esperados:

```powershell
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --detector research

.\scripts\python.bat scripts\benchmark_vision.py --suite full --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite full --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite full --detector research
```

### Metricas

Medir:

- recall;
- precisao;
- F1;
- IoU medio;
- recall por tamanho de face;
- recall por iluminacao;
- recall por blur;
- recall por oclusao;
- FPS medio;
- FPS minimo;
- falsos negativos mais importantes;
- falsos positivos mais importantes.

### Saidas

Gerar em:

```text
results/benchmark/
```

Arquivos:

- `summary.md`
- `summary.html`
- `detections.csv`
- `metrics_by_detector.csv`
- `metrics_by_face_size.csv`
- `false_negatives_contact_sheet.png`
- `false_positives_contact_sheet.png`

## Cenarios De Teste

### Individual

Testar:

- rosto perto;
- rosto medio;
- rosto distante;
- rosto lateral;
- rosto parcialmente fora do frame;
- baixa luz;
- luz forte;
- blur de movimento;
- oclusao parcial;
- webcam real no quarto;
- video local conhecido.

### Plateia

Testar:

- multiplas faces grandes;
- multiplas faces pequenas;
- sala de aula distante;
- pessoas parcialmente viradas;
- pessoas no fundo;
- faces sobrepostas;
- baixa iluminacao;
- video livre baixado;
- cenas sem rosto para medir falso positivo.

## Criterios De Aceitacao

Uma mudanca so deve ser aceita se:

- recall em faces pequenas/distantes melhorar pelo menos 20 pontos percentuais contra o baseline `mediapipe`;
- recall em faces perto/medio nao cair mais que 2 pontos percentuais;
- precisao geral ficar em pelo menos 85%;
- Modo Individual em 960x540 manter pelo menos 15 FPS medio;
- Modo Plateia com `--max-faces 24` manter pelo menos 8 FPS medio em 1080p e 10 FPS medio em 960x540;
- nao houver crash em videos longos, cenas sem rosto, oclusao parcial ou baixa luz;
- `pytest`, `ruff`, `compileall` e `presentation_check.py --skip-slow` continuarem passando.

Limiares escolhidos para o perfil padrao: Individual `0.70`; Plateia `0.60` com `max_faces=24`. No sweep em videos 1080p completos, `0.50` comecou a supercontar em cenas com multidao; `0.60` manteve melhor erro medio na Plateia.

## Estado Implementado E Evidencia Medida

Implementado:

- perfil `enhanced` padrao com YuNet, NMS, recortes ampliados e MediaPipe;
- perfil `research` opcional com SCRFD/ONNX Runtime;
- Control Center com selecao de perfil, download de corpus e execucao de benchmark;
- downloader com manifesto, hash e fontes para WIDER FACE, videos publicos e modelos;
- benchmark com variantes de luz, blur, oclusao, distancia e cena sem rosto;
- relatorio com recall de face, recall de landmarks, precisao, F1, IoU, FPS e contact sheets;
- Modo Plateia separa faces localizadas de faces com pose valida.

Resultados smoke reproduzidos com o perfil padrao `enhanced` e a melhor configuracao por modo:

| Modo | Baseline recall face | Enhanced recall face | Enhanced recall landmarks | Precisao enhanced | FPS 960x540 enhanced | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Individual | 0.193 | 0.592 | 0.294 | 0.894 | 29.0 | PASS |
| Plateia | 0.011 | 0.443 | 0.089 | 0.851 | 74.5 | PASS |

Comparativo smoke dos perfis principais:

| Modo | Perfil | Recall face | Recall landmarks | Precisao | F1 | FPS 960x540 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Individual | `mediapipe` | 0.193 | 0.193 | 1.000 | 0.324 | 242.4 |
| Individual | `enhanced` | **0.592** | **0.294** | 0.894 | **0.712** | 29.0 |
| Plateia | `mediapipe` | 0.011 | 0.011 | 1.000 | 0.022 | 491.6 |
| Plateia | `enhanced` | **0.443** | **0.089** | 0.851 | **0.583** | **74.5** |

O perfil `enhanced` foi escolhido porque superou o baseline em recall e F1 nos dois modos, mantendo FPS adequado para apresentacao. O recall de landmarks demonstra o limite remanescente: nem toda face distante permite pose confiavel. Por isso o produto exibe essas deteccoes, mas nao as converte em engajamento.

### Auditoria HQ 1080p Em Video Inteiro

Depois do benchmark sintetico/WIDER, adicionamos uma auditoria manual em tres MP4 1080p. Cada perfil processa o video do primeiro ao ultimo frame, e os frames contados manualmente sao comparados contra o log temporal real.

| Perfil | Acertos | <=1 rosto | Erro medio | Manual dentro da janela +/-1s | FPS medio 1080p |
| --- | ---: | ---: | ---: | ---: | ---: |
| `enhanced_c60_m24` | 5/11 | 10/11 | 0.636 | 10/11 | 13.3 |
| `enhanced_m24` | 5/11 | 9/11 | 0.727 | 8/11 | 14.7 |
| `enhanced_c50_m24` | 4/11 | 9/11 | 0.818 | 10/11 | 13.7 |
| `research_c55_m24` | 5/11 | 8/11 | 1.364 | 6/11 | 28.4 |

Decisao aplicada ao app: Individual usa `--face-confidence 0.70`; Plateia usa `--max-faces 24` e `--face-confidence 0.60` por padrao. O teto antigo de 8 faces era o principal limitador em sala cheia.

## Gaps Que O Benchmark Deve Identificar

### Modo Individual

Possiveis gaps:

- face pequena demais para MediaPipe;
- crop ruim causando landmarks instaveis;
- baseline calibrado com face mal detectada;
- yaw/pitch instaveis em baixa resolucao;
- alertas acionando por falha de landmark;
- perda de face ao inclinar a cabeca;
- FPS caindo com detector pesado.

### Modo Plateia

Possiveis gaps:

- subcontagem de faces distantes;
- faces pequenas sem landmarks suficientes para pose;
- calibracao de palco contaminada por pessoas olhando para lados diferentes;
- percentual de engajamento punindo faces detectadas mas sem pose valida;
- duplicatas de face no mesmo frame;
- falsos positivos em fundo da sala.

## Regras De Implementacao

- Nao treinar modelo proprio.
- Nao substituir `solvePnP` nesta etapa.
- Nao tornar SCRFD obrigatorio para abrir o app.
- Nao commitar datasets, videos ou modelos grandes.
- Documentar claramente qualquer modelo com licenca de pesquisa.
- Preservar o fluxo de instalacao: clone, `cd`, `.\run.bat` no Windows ou `./run.sh` no Linux/macOS.

## Testes De Regressao Obrigatorios

Antes de aceitar a implementacao:

```powershell
.\run.bat -SetupOnly -SkipModel
.\scripts\python.bat -m ruff check src tests scripts
.\scripts\python.bat -m pytest -q
.\scripts\python.bat -m compileall src tests scripts
.\scripts\python.bat scripts\presentation_check.py --skip-slow
```

Equivalente Linux/macOS:

```bash
./run.sh --setup-only --skip-model
./scripts/python.sh -m ruff check src tests scripts
./scripts/python.sh -m pytest -q
./scripts/python.sh -m compileall src tests scripts
./scripts/python.sh scripts/presentation_check.py --skip-slow
```

Depois:

```powershell
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --detector research
```

Se houver tempo de bancada, o runner tambem aceita suite `full`; para a entrega, a evidencia principal e o smoke com a configuracao atual mais a auditoria HQ em videos completos:

```powershell
.\scripts\python.bat scripts\benchmark_vision.py --suite full --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite full --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite full --detector research
```

## Referencias

- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- OpenCV YuNet: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md
- InsightFace Model Zoo: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- WIDER FACE paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf
