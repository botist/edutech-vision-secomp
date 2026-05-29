# Resultados De Benchmark - EduTech Vision

## Objetivo

Avaliar se o detector padrao `enhanced` (OpenCV YuNet + landmarks MediaPipe em recortes) melhora a localizacao de rostos pequenos e distantes sem comprometer a demonstracao em tempo real.

## Metodologia

- Dataset anotado: WIDER FACE validation, com variantes sinteticas de iluminacao, blur, distancia e oclusao.
- Desempenho: videos publicos de sala de aula em 960x540.
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

## Reproducao

```powershell
.\run.bat -SetupOnly
.\scripts\python.bat scripts\download_benchmarks.py
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector mediapipe
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode individual --detector enhanced
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector mediapipe --max-faces 24
.\scripts\python.bat scripts\benchmark_vision.py --suite smoke --mode plateia --detector enhanced --max-faces 24
```

No Linux/macOS, use `./run.sh --setup-only` e `./scripts/python.sh` nos mesmos scripts.

O plano completo, as fontes e a interpretacao das limitacoes estao em `docs/PLANO_TESTES_PRIOR_ART.md`.
