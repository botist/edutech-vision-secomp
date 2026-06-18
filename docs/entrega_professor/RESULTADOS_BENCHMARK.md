# Resultados De Benchmark - EduTech Vision

## Objetivo

Avaliar se o detector padrao `enhanced` (OpenCV YuNet + landmarks MediaPipe em recortes) melhora a localizacao de rostos pequenos e distantes sem comprometer a demonstracao em tempo real.

## Metodologia

- Dataset anotado: WIDER FACE validation, com variantes sinteticas de iluminacao, blur, distancia e oclusao.
- Desempenho: videos publicos de sala de aula em 960x540.
- Modo Individual: videos frontais 1080p completos e retratos estaticos, baixados localmente por manifesto reproduzivel.
- Perfis comparados: `mediapipe` (baseline), `enhanced` (YuNet, padrao) e `research` (SCRFD).
- Configuracao atual: Individual `--face-confidence 0.70`; Plateia `--face-confidence 0.60 --max-faces 24`.
- Relatorio gerado localmente: `results/benchmark/summary.html`.

`Recall face` mede uma caixa detectada corretamente. `Recall landmarks` mede somente as faces que tambem geram pontos suficientes para EAR, MAR ou pose. No app, uma face localizada sem landmarks confiaveis aparece na interface, mas nao influencia engajamento.

## Benchmark Smoke Com Configuracao Atual

| Modo | Perfil | Recall face | Recall landmarks | Precisao | F1 | FPS 960x540 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Individual | `mediapipe` | 0.193 | 0.193 | 1.000 | 0.324 | 242.4 |
| Individual | `enhanced` | **0.592** | **0.294** | 0.894 | **0.712** | 29.0 |
| Plateia | `mediapipe` | 0.011 | 0.011 | 1.000 | 0.022 | 491.6 |
| Plateia | `enhanced` | **0.443** | **0.089** | 0.851 | **0.583** | **74.5** |

## Criterios E Decisao

| Modo | Ganho de recall do `enhanced` | Precisao minima | FPS minimo exigido | Resultado |
| --- | ---: | ---: | ---: | --- |
| Individual | +0.399 | 0.85 | 15 | PASS |
| Plateia | +0.432 | 0.85 | 10 | PASS |

O perfil `enhanced` e o padrao da apresentacao: obteve maior recall e maior F1 nos dois modos que o baseline, mantendo FPS suficiente. Em auditoria manual com videos 1080p completos, o melhor ajuste foi `enhanced_c60_m24`: erro medio 0.636 rosto e 10/11 frames com erro maximo de 1 rosto. O perfil `research` permanece somente como comparacao academica, inclusive porque os pesos SCRFD/InsightFace sao disponibilizados para uso non-commercial/research.

## Auditoria Especifica Do Modo Individual

Depois dos testes de detector, o Modo Individual foi auditado no pipeline completo: face, landmarks, EAR/MAR, pose, baseline, limiares e classificacao temporal. O auditor processa videos inteiros, redimensionados para a mesma resolucao do app (`960x540`), e registra uma linha por frame.

Resultado consolidado em tres videos frontais reais:

| Midia | Frames | Face/metricas validas | Frontal | Expressao neutra | Passou |
| --- | ---: | ---: | ---: | ---: | ---: |
| Videos frontais 1080p completos | 15.193 | 86,6% | 82,5% | 76,7% | 73,0% |

Por arquivo:

| Video | Frames | Face/metricas validas | Frontal | Passou |
| --- | ---: | ---: | ---: | ---: |
| `lakhan_lal_interview_1080p.webm` | 1.345 | 100,0% | 95,7% | 73,8% |
| `nasa_cristian_parker_interview_1080p.webm` | 8.427 | 85,8% | 79,4% | 74,1% |
| `nasa_jamie_brock_spherex_1080p.webm` | 5.421 | 84,6% | 83,9% | 71,1% |

Os frames sem face dos videos NASA correspondem principalmente a cartelas e b-roll de abertura, nao a falha de rastreamento sobre pessoa frontal. A auditoria tambem revelou e corrigiu dois problemas reais: `pitch` perto de `+179/-179` era tratado como postura extrema, e a calibracao podia terminar antes da primeira face valida em videos com abertura sem rosto.

## Reproducao

```powershell
.\run.bat -SetupOnly
.\scripts\python.bat scripts\download_benchmarks.py
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector mediapipe --max-faces 24
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector enhanced --max-faces 24
.\scripts\python.bat scripts\download_individual_benchmarks.py
.\scripts\python.bat scripts\audit_individual_media.py --media-dir assets\benchmarks\individual --output-dir results\individual_media_audit --detector enhanced --confidence 0.70 --frame-step 1 --max-video-seconds 0
```

No Linux/macOS, use `./run.sh --setup-only` e `./scripts/python.sh` nos mesmos scripts.

O plano completo, as fontes e a interpretacao das limitacoes estao em `docs/PLANO_TESTES_PRIOR_ART.md`.
