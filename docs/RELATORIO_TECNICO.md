# Relatorio Tecnico - EduTech Vision

## 1. Objetivo

Desenvolver um produto de PDI para monitoramento educacional em dois cenarios: estudo individual por webcam e analise agregada de plateia. O foco tecnico e transformar sinais visuais de face e cabeca em indicadores temporais de atencao, postura e engajamento.

## 2. Arquitetura

```text
Captura resiliente
  -> YuNet (padrao) ou SCRFD (pesquisa) detecta faces
  -> MediaPipe Face Landmarker em crops ampliados
    -> Modo Individual: EAR, MAR, solvePnP, suavizacao, estados sustentados
    -> Modo Plateia: solvePnP multi-face, classificacao por direcao, agregacao 10s
  -> Interface OpenCV
  -> Logs CSV e relatorio visual
```

Modulos principais:

- `core/detection.py`: detectores YuNet/SCRFD, NMS e recortes de face.
- `core/landmarks.py`: MediaPipe Face Landmarker aplicado ao frame ou aos recortes.
- `core/metrics.py`: calculo de EAR e MAR.
- `core/pose.py`: estimativa de yaw, pitch e roll com `cv2.solvePnP`.
- `core/filters.py`: media movel e condicoes sustentadas.
- `core/aggregation.py`: janela temporal de engajamento para plateia.
- `utils/camera.py`: reconexao amigavel da webcam.
- `modes/individual.py` e `modes/audience.py`: logica dos produtos.
- `modes/demo.py`: modo sintetico de contingencia com visual completo.
- `run.bat`/`run.sh`: bootstrap em um comando para Windows e Linux/macOS.
- `scripts/download_models.py`: baixa o modelo oficial `face_landmarker.task` necessario para a API nova do MediaPipe.
- `scripts/download_benchmarks.py`: baixa WIDER FACE, videos livres e manifesto de fontes/licencas.
- `scripts/benchmark_vision.py`: compara baseline, YuNet e SCRFD com metricas e exemplos de erro.
- `scripts/generate_session_report.py`: gera Markdown, HTML, PDF e graficos agregados a partir dos CSVs numericos.

## 3. Modo Individual

O sistema detecta uma face, calcula EAR e MAR, estima pose da cabeca e suaviza os sinais por media movel. A calibracao usa os primeiros 5 segundos a partir da primeira amostra facial valida, nao simplesmente a partir da abertura do app. Isso evita baseline vazio quando um video ou webcam inicia com tela sem rosto.

Alertas:

- fadiga por olho fechado sustentado: EAR abaixo do limiar dinamico;
- bocejo: MAR acima do limiar dinamico;
- queda postural: diferenca de pitch acima de tolerancia;
- desatencao: diferenca de yaw acima de tolerancia.

Os alertas exigem persistencia temporal para evitar falsos positivos causados por piscadas ou movimentos rapidos. O baseline de yaw/pitch usa estatistica circular, porque em solvePnP uma face frontal pode aparecer perto de `+179` ou `-179` graus; esses valores sao praticamente equivalentes e nao devem ser tratados como extremos opostos. Para fadiga ocular, o alerta exige queda simultanea do EAR medio e dos dois olhos, reduzindo falso positivo de um landmark instavel.

## 4. Modo Plateia

O sistema detecta multiplas faces e estima yaw/pitch para cada uma. Nos primeiros segundos, o eixo do palco e calibrado pela mediana das poses observadas; depois, cada face e classificada como "olhando para o palco" quando yaw e pitch ficam dentro das tolerancias configuradas em relacao a esse eixo. A saida cientifica e agregada por janela de 10 segundos.

Faces distantes que o detector encontra, mas nao oferecem landmarks estaveis, sao exibidas e contabilizadas separadamente como deteccao sem pose. Elas nao entram no percentual de direcao da cabeca.

Telemetria registrada:

- timestamp;
- media de faces detectadas;
- media de faces atentas;
- percentual de engajamento;
- FPS medio.

## 5. Algoritmos de PDI usados

- Landmarks faciais com MediaPipe Face Landmarker.
- EAR e MAR calculados por distancias euclidianas entre landmarks.
- Pose de cabeca com modelo facial 3D generico e `cv2.solvePnP`.
- Suavizacao temporal por media movel.
- Maquina de estados para alertas sustentados.
- Agregacao estatistica em janela temporal.

## 6. Protocolos de avaliacao

Os scripts em `tests/` executam os cinco protocolos exigidos:

- `test_lighting.py`: coleta em iluminacao baixa, media e alta.
- `test_occlusion.py`: mede recuperacao apos oclusao de 3 segundos.
- `test_fps.py`: gera telemetria de FPS em janela de 60 segundos.
- `test_confusion_matrix.py`: calcula matriz de confusao a partir de CSV rotulado.
- `test_failover.py`: verifica desconexao e reconexao da webcam sem encerramento abrupto.
- `benchmark_vision.py`: mede recall/precisao/F1 no WIDER FACE e FPS em videos reais, comparando os perfis de detector.
- No modo plateia, a configuracao padrao usa uma passagem YuNet por frame: no benchmark ela preservou o ganho de recall e eliminou o custo instavel da segunda escala.

No benchmark smoke com a configuracao atual, o perfil `enhanced` obteve recall de face `0.592` no Individual e `0.443` na Plateia, contra `0.193` e `0.011` do baseline. A precisao foi `0.894`/`0.851`, e o FPS em 960x540 foi `29.0`/`74.5`. A auditoria HQ em videos completos confirmou o melhor ajuste de Plateia como `enhanced_c60_m24`, com erro medio de `0.636` rosto e 10/11 casos com erro de no maximo 1 rosto. O relatorio tambem separa recall de landmarks (`0.294`/`0.089`), pois uma caixa distante nao implica pose valida.

Para o pipeline final do Modo Individual, uma auditoria separada processou videos frontais 1080p completos, redimensionados para a mesma resolucao operacional do app. Foram `15.193` frames: `86,6%` com face e metricas validas, `82,5%` classificados como frontais e deltas medianos de `4,0 deg` em yaw e `1,8 deg` em pitch. As principais falhas restantes nesses videos sao cartelas/b-roll sem pessoa no quadro e expressoes nao neutras, nao perda sistematica de rosto frontal.

## 7. Limitacoes

- A estimativa de pose depende de camera frontal e iluminacao suficiente.
- O Modo Plateia usa classificacao por limiares, nao reconhecimento de identidade.
- Faces pequenas podem ser detectadas sem pose valida; o painel mostra essa diferenca em vez de inferir direcao sem evidência.
- Resultados finais do artigo devem ser obtidos com amostras reais coletadas pela equipe.

## 8. Limitacoes e uso responsavel

O sistema gera indicadores visuais aproximados. Ele nao deve ser apresentado como diagnostico clinico, avaliacao pedagogica definitiva ou medida individual de aprendizagem. Iluminacao, distancia, resolucao, FPS e postura inicial da camera influenciam diretamente o resultado.

O Modo Demo e separado dos modos reais e deixa claro quando a cena e sintetica. Ele existe para contingencia de apresentacao e para explicar o fluxo visual quando a camera nao estiver disponivel.
