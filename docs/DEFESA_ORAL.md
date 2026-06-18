# Defesa Oral - Perguntas Provaveis

## Perguntas tecnicas

**Como o EAR e calculado?**  
Usamos seis landmarks por olho. A soma de duas distancias verticais e dividida por duas vezes a distancia horizontal. Quando o olho fecha, as distancias verticais caem e o EAR diminui.

**Como o MAR e calculado?**  
Usamos cantos da boca e pontos dos labios. A abertura vertical e dividida pela largura da boca. Bocejo tende a aumentar o MAR de forma sustentada.

**Por que usar `solvePnP`?**  
Porque temos pontos 2D detectados na imagem e um modelo 3D aproximado da face. O `solvePnP` estima a rotacao da cabeca, permitindo obter yaw, pitch e roll.

**Por que media movel?**  
Para reduzir falsos positivos causados por piscadas, pequenas oscilacoes de landmark e ruido de deteccao frame a frame.

**Por que maquina de estados?**  
Porque o alerta so faz sentido quando a condicao persiste por um periodo minimo. Uma piscada isolada nao deve virar fadiga.

**Qual e o core de PDI do projeto?**  
Detector YuNet para localizar faces, landmarks MediaPipe nos recortes, metricas geometricas EAR/MAR, pose por `solvePnP`, suavizacao temporal e agregacao estatistica.

## Perguntas de metodologia

**Como validaram acuracia?**  
Com WIDER FACE para deteccao anotada, benchmark smoke reproduzivel, auditoria HQ em videos 1080p completos e uma auditoria propria do modo Individual em videos frontais completos. Comparamos o baseline MediaPipe contra YuNet e tambem validamos o pipeline final de face, landmarks, EAR/MAR, pose e baseline.

**Qual foi o resultado mensurado?**
No smoke benchmark com a configuracao atual, recall de face do Individual subiu de `0.193` para `0.592`; na Plateia, de `0.011` para `0.443`. A precisao enhanced ficou em `0.894` e `0.851`, com `29.0` e `74.5 FPS`. Na auditoria HQ de Plateia, `enhanced_c60_m24` ficou com erro medio de `0.636` rosto e 10/11 casos com erro de no maximo 1 rosto. Na auditoria especifica do Individual, foram `15.193` frames de videos frontais completos, com `86,6%` de frames com face/metricas validas, `82,5%` frontais e deltas medianos de `4,0 deg` em yaw e `1,8 deg` em pitch.

**O que melhorou no modo Individual alem do detector?**
A calibracao agora espera a primeira face valida antes de contar os 5 segundos; o baseline usa media circular para angulos, evitando erro quando pitch aparece como `+179/-179`; os limiares de EAR/MAR sao adaptados pelo baseline; e o alerta de olho fechado exige ambos os olhos fechados para reduzir falso positivo de piscada ou landmark instavel.

**Por que usar YuNet antes do MediaPipe?**
O MediaPipe continua excelente para landmarks, mas perdia rostos pequenos no frame completo. YuNet encontra a regiao da face; ao aplicar os landmarks no recorte ampliado, recuperamos rostos distantes sem treinar modelo proprio.

**Por que nao usar duas escalas YuNet em toda a plateia?**
O benchmark mostrou ganho pequeno de recall com a segunda escala e custo alto de FPS. A demonstracao usa uma passagem por frame para manter responsividade, preservando o ganho relevante sobre o baseline.

**O que e o perfil `research`?**
E uma comparacao opcional com SCRFD/InsightFace, prior art forte para deteccao. Mantivemos fora do caminho padrao porque o melhor perfil distribuivel nos testes foi YuNet `enhanced`, sem peso academico opcional e com instalacao mais simples.

**Como testaram iluminacao?**  
Executando o sistema por 60 segundos em tres niveis: baixa, media e alta iluminacao. O criterio e nao crashar e nao degradar mais que 30% em relacao a condicao ideal.

**Como testaram oclusao?**  
Bloqueando a face por 3 segundos e medindo se a deteccao retorna em ate 2 segundos apos a reapresentacao.

## Perguntas de escopo

**O sistema identifica pessoas?**  
Nao. O modo plateia estima direcao de cabeca e calcula percentual de engajamento por janela.

**Isso diagnostica fadiga clinica?**  
Nao. O sistema e uma demonstracao academica de PDI e gera indicadores visuais aproximados, nao diagnostico medico ou psicologico.

**O que e o Modo Demo?**
E uma simulacao visual sem camera para contingencia de apresentacao e explicacao do fluxo quando nao houver webcam disponivel.

## Trechos de codigo para saber explicar

- `core/metrics.py`: formulas EAR/MAR.
- `core/pose.py`: `solvePnP`, matriz de camera e pontos 3D.
- `core/filters.py`: media movel e condicao sustentada.
- `core/detection.py`: YuNet/SCRFD, NMS e recortes.
- `modes/individual.py`: baseline, limiares e alertas.
- `modes/audience.py`: calibracao do palco e agregacao temporal.
- `utils/camera.py`: reconexao de webcam.
