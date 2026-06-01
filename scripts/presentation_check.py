from __future__ import annotations

import argparse
import csv
import importlib.metadata
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "presentation_check_report.md"
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_TRACKED_MEDIA = {
    "docs/entrega_professor/poster_secomp_grupo3.png",
}
ALLOWED_MEDIA_PREFIXES: tuple[str, ...] = ()
ALLOWED_RESULTS_MEDIA = {
    "confusion_matrix.png",
    "session_summary_dashboard.png",
    "session_individual_timeline.png",
    "session_audience_timeline.png",
    "session_alerts.png",
    "false_negatives_contact_sheet.png",
    "false_positives_contact_sheet.png",
}
REQUIRED_DOCS = [
    "README.md",
    "docs/RUBRICA_MAX_NOTA.md",
    "docs/DEFESA_ORAL.md",
    "docs/EXPOSICAO_SECOMP.md",
    "docs/CHECKLIST_SOFTWARE_APRESENTACAO.md",
    "docs/RELATORIO_TECNICO.md",
    "docs/PLANO_TESTES_PRIOR_ART.md",
    "docs/entrega_professor/README.md",
    "docs/entrega_professor/RESULTADOS_BENCHMARK.md",
    "docs/entrega_professor/artigo_edutech_vision_grupo3.pdf",
    "docs/entrega_professor/artigo_edutech_vision_grupo3.docx",
    "docs/entrega_professor/poster_secomp_grupo3.pdf",
    "docs/entrega_professor/poster_secomp_grupo3.png",
]
PROTOCOL_FILES = [
    "results/fps_log.csv",
    "results/lighting_evaluation.csv",
    "results/occlusion_recovery.csv",
    "results/confusion_metrics.csv",
    "results/confusion_matrix.png",
    "results/failover_log.csv",
    "results/protocol_summary.md",
    "results/presentation_check_report.md",
    "results/session_report.md",
    "results/session_report.html",
    "results/session_report.pdf",
    "results/benchmark/summary.html",
    "results/report_charts/session_summary_dashboard.png",
    "results/report_charts/session_individual_timeline.png",
    "results/report_charts/session_audience_timeline.png",
    "results/report_charts/session_alerts.png",
]
CODE_EXPLANATION_FILES = [
    "src/edutech_vision/core/metrics.py",
    "src/edutech_vision/core/pose.py",
    "src/edutech_vision/core/filters.py",
    "src/edutech_vision/core/aggregation.py",
    "src/edutech_vision/core/landmarks.py",
    "src/edutech_vision/core/detection.py",
    "src/edutech_vision/modes/individual.py",
    "src/edutech_vision/modes/audience.py",
    "src/edutech_vision/modes/demo.py",
    "src/edutech_vision/launcher.py",
    "src/edutech_vision/utils/camera.py",
    "src/edutech_vision/protocols.py",
    "scripts/benchmark_vision.py",
    "scripts/download_benchmarks.py",
]


@dataclass(frozen=True)
class CheckResult:
    section: str
    item: str
    status: str
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight da checklist de software da apresentacao.")
    parser.add_argument("--skip-slow", action="store_true", help="Pula ruff, pytest e compileall.")
    parser.add_argument("--strict", action="store_true", help="Retorna erro tambem quando houver avisos.")
    parser.add_argument("--camera-check", action="store_true", help="Inclui teste sem janela para abrir a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam usado com --camera-check.")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    checks: list[CheckResult] = []
    checks.extend(check_repository())
    checks.extend(check_environment(camera_check=args.camera_check, camera=args.camera))
    checks.extend(check_validation(skip_slow=args.skip_slow))
    checks.extend(check_evidence_files())
    checks.extend(check_protocol_readiness())
    checks.extend(check_showcase_focus())
    checks.extend(check_performance_defaults())
    checks.extend(check_code_explanation_files())
    checks.extend(manual_demo_checks())

    write_report(args.output, checks)
    print_summary(checks, args.output)

    has_failures = any(check.status == "FAIL" for check in checks)
    has_warnings = any(check.status == "WARN" for check in checks)
    raise SystemExit(1 if has_failures or (args.strict and has_warnings) else 0)


def check_repository() -> list[CheckResult]:
    section = "1. Estado do repositorio"
    checks: list[CheckResult] = []

    status = run(["git", "status", "--porcelain"])
    tracked_changes = [
        line
        for line in status.stdout.splitlines()
        if line
        and not line.startswith("!!")
        and not line.endswith("results/presentation_check_report.md")
        and line.strip() != "?? AGENTS.md"
    ]
    checks.append(
        result(
            section,
            "`git status --short` nao mostra mudancas inesperadas em arquivos rastreados.",
            not tracked_changes,
            "limpo" if not tracked_changes else "; ".join(tracked_changes[:8]),
        )
    )

    log = run(["git", "log", "--oneline", "--decorate", "-1"])
    checks.append(result(section, "Existe commit recente com a versao que sera apresentada.", log.ok, log.stdout.strip()))

    tags = run(["git", "tag", "--list"])
    checks.append(result(section, "Existe tag de marco.", bool(tags.stdout.strip()), tags.stdout.strip() or "nenhuma tag encontrada"))

    for doc in REQUIRED_DOCS:
        checks.append(result(section, f"`{doc}` existe.", (ROOT_DIR / doc).exists(), doc))

    media = tracked_media_files()
    risky_media = [
        path
        for path in media
        if path not in ALLOWED_TRACKED_MEDIA and not path.startswith(ALLOWED_MEDIA_PREFIXES)
    ]
    checks.append(
        result(
            section,
            "Repositorio nao tem midia bruta pesada versionada por acidente.",
            not risky_media,
            "nenhum arquivo de midia rastreado" if not risky_media else "; ".join(risky_media),
        )
    )

    model_path = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
    yunet_path = ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    model_ignored = run(["git", "check-ignore", "assets/models/face_landmarker.task"]).ok
    yunet_ignored = run(["git", "check-ignore", "assets/models/face_detection_yunet_2023mar.onnx"]).ok
    checks.append(result(section, "`face_landmarker.task` existe localmente.", model_path.exists(), str(model_path)))
    checks.append(result(section, "`face_landmarker.task` continua ignorado pelo Git.", model_ignored, "git check-ignore"))
    checks.append(result(section, "`face_detection_yunet_2023mar.onnx` existe localmente.", yunet_path.exists(), str(yunet_path)))
    checks.append(result(section, "Modelo YuNet continua ignorado pelo Git.", yunet_ignored, "git check-ignore"))
    checks.append(check_results_are_numeric_or_aggregate())
    return checks


def check_environment(camera_check: bool, camera: int) -> list[CheckResult]:
    section = "2. Ambiente Python"
    checks: list[CheckResult] = []

    checks.append(result(section, "Ambiente `.venv` existe.", (ROOT_DIR / ".venv").exists(), str(ROOT_DIR / ".venv")))
    checks.append(
        result(
            section,
            "Python do ambiente esta em 3.11.",
            sys.version_info[:2] == (3, 11),
            sys.version.split()[0],
        )
    )
    checks.append(check_requirements_pinned())
    checks.append(check_requirements_match_pyproject())
    checks.append(check_installed_editable())

    help_result = run([sys.executable, "-m", "edutech_vision", "--help"])
    checks.append(result(section, "`python -m edutech_vision --help` funciona.", help_result.ok, first_line(help_result.stdout, help_result.stderr)))
    checks.append(result(section, "CLI oferece modo visual `--showcase`.", help_result.ok and "--showcase" in help_result.stdout, "flag --showcase"))
    checks.append(result(section, "CLI oferece modo demo sintetico.", help_result.ok and "demo" in help_result.stdout, "modo demo"))
    checks.append(result(section, "CLI oferece perfis `mediapipe`, `enhanced` e `research`.", help_result.ok and "--detector" in help_result.stdout, "flag --detector"))
    checks.append(result(section, "`run.bat` existe.", (ROOT_DIR / "run.bat").exists(), "bootstrap em 1 comando"))
    checks.append(result(section, "`run.sh` existe.", (ROOT_DIR / "run.sh").exists(), "bootstrap Linux/macOS em 1 comando"))
    checks.append(result(section, "Modulo do Control Center importa.", can_import_launcher(), "edutech_vision.launcher"))

    doctor = run([sys.executable, "scripts/doctor.py"])
    checks.append(result(section, "`scripts/doctor.py` passa sem erros.", doctor.ok, first_line(doctor.stdout, doctor.stderr)))

    model_path = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
    checks.append(result(section, "O modelo MediaPipe foi baixado.", model_path.exists() and model_path.stat().st_size > 0, str(model_path)))
    yunet_path = ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    checks.append(result(section, "O modelo YuNet do perfil `enhanced` foi baixado.", yunet_path.exists() and yunet_path.stat().st_size > 0, str(yunet_path)))
    if camera_check:
        doctor_camera = run([sys.executable, "scripts/doctor.py", "--camera-check", "--camera", str(camera)])
        checks.append(result(section, f"Webcam {camera} abre e retorna frames.", doctor_camera.ok, last_relevant_line(doctor_camera.stdout, doctor_camera.stderr)))
    else:
        checks.append(manual(section, "Webcam abre e retorna frames.", "rodar presentation_check.py --camera-check no notebook da apresentacao"))
    return checks


def check_validation(skip_slow: bool) -> list[CheckResult]:
    section = "3. Validacao tecnica antes de apresentar"
    checks: list[CheckResult] = []

    if skip_slow:
        checks.extend(
            manual(section, item, "pulou por --skip-slow")
            for item in ["`ruff` passa.", "`pytest` passa.", "`compileall` passa."]
        )
    else:
        if module_available("ruff"):
            ruff = run([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"])
            checks.append(result(section, "`ruff` passa.", ruff.ok, first_line(ruff.stdout, ruff.stderr)))
        else:
            checks.append(manual(section, "`ruff` passa.", "instalar ferramentas dev: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt"))

        if module_available("pytest"):
            pytest = run([sys.executable, "-m", "pytest", "-q"])
            checks.append(result(section, "`pytest` passa.", pytest.ok, first_line(pytest.stdout, pytest.stderr)))
        else:
            checks.append(manual(section, "`pytest` passa.", "instalar ferramentas dev: .\\scripts\\python.bat -m pip install -r requirements-dev.txt"))

        compileall = run([sys.executable, "-m", "compileall", "src", "tests", "scripts"])
        checks.append(result(section, "`compileall` passa.", compileall.ok, "compileall ok" if compileall.ok else first_line(compileall.stdout, compileall.stderr)))

    detector = run(
        [
            sys.executable,
            "-c",
            "from edutech_vision.core.landmarks import FaceLandmarkDetector; "
            "d=FaceLandmarkDetector(max_faces=1, detector_profile='enhanced'); d.close(); print('detector_ok')",
        ]
    )
    checks.append(result(section, "Detector padrao `enhanced` inicializa.", detector.ok, first_line(detector.stdout, detector.stderr)))

    labels = ROOT_DIR / "assets" / "samples" / "labels_real.csv"
    if labels.exists() and count_csv_rows(labels) >= 50:
        command_labels = "assets/samples/labels_real.csv"
    else:
        command_labels = "assets/samples/labels_demo.csv"
    confusion = run([sys.executable, "tests/test_confusion_matrix.py", "--labels", command_labels, "--output-dir", "results"])
    checks.append(result(section, "Matriz de confusao demo ou real gera saida sem erro.", confusion.ok, command_labels))

    session_report = run([sys.executable, "scripts/generate_session_report.py"])
    checks.append(result(section, "Relatorio de sessao e gerado.", session_report.ok, first_line(session_report.stdout, session_report.stderr)))

    checks.append(manual(section, "O app abre em Modo Individual.", "executar comando da checklist com webcam e janela OpenCV"))
    checks.append(manual(section, "O app abre em Modo Plateia.", "executar comando da checklist com webcam e janela OpenCV"))
    return checks


def check_evidence_files() -> list[CheckResult]:
    section = "6. Evidencias que devem estar prontas para abrir"
    checks: list[CheckResult] = []
    for path in PROTOCOL_FILES:
        full_path = ROOT_DIR / path
        status = "OK" if full_path.exists() else "WARN"
        detail = "existe" if full_path.exists() else "pendente de coleta real"
        checks.append(CheckResult(section, f"`{path}`", status, detail))
    for path in [
        "docs/RELATORIO_TECNICO.md",
        "docs/entrega_professor/artigo_edutech_vision_grupo3.pdf",
        "docs/entrega_professor/artigo_edutech_vision_grupo3.docx",
        "docs/entrega_professor/poster_secomp_grupo3.pdf",
        "docs/entrega_professor/poster_secomp_grupo3.png",
    ]:
        checks.append(result(section, f"`{path}`", (ROOT_DIR / path).exists(), "existe"))
    benchmark_summary = RESULTS_DIR / "benchmark" / "summary.md"
    benchmark_text = benchmark_summary.read_text(encoding="utf-8") if benchmark_summary.exists() else ""
    checks.append(
        result(
            section,
            "Benchmark smoke do perfil `enhanced` possui gates aprovados nos dois modos.",
            benchmark_text.count("| PASS |") >= 2,
            "dois gates smoke aprovados" if benchmark_text.count("| PASS |") >= 2 else "rode benchmarks smoke enhanced/baseline nos dois modos",
        )
    )
    return checks


def check_protocol_readiness() -> list[CheckResult]:
    section = "7. Protocolos ao vivo"
    checks = [
        manual(section, "Executar baixa, media e alta iluminacao.", "requer webcam e mudanca fisica da iluminacao"),
        manual(section, "Confirmar que nao houve crash.", "registrar durante protocolo ao vivo"),
        check_lighting_csv(),
        manual(section, "Explicar degradacao contra condicao ideal.", "usar resultados em protocol_summary.md"),
        manual(section, "Registrar baseline de oclusao.", "requer execucao ao vivo"),
        manual(section, "Bloquear a face por 3 segundos.", "acao fisica durante avaliacao"),
        manual(section, "Remover bloqueio.", "acao fisica durante avaliacao"),
        check_occlusion_csv(),
        manual(section, "Rodar FPS por 60 segundos.", "requer webcam no notebook da apresentacao"),
        check_fps_csv(),
        check_real_labels(),
        check_confusion_metrics(),
        manual(section, "Mostrar matriz completa.", "abrir results/confusion_matrix.png"),
        manual(section, "Rodar failover.", "requer desconectar/reconectar webcam"),
        manual(section, "Desconectar webcam.", "acao fisica durante avaliacao"),
        manual(section, "Reconectar webcam.", "acao fisica durante avaliacao"),
        check_failover_csv(),
    ]
    return checks


def check_showcase_focus() -> list[CheckResult]:
    section = "8. Impacto visual e escopo tecnico"
    checks: list[CheckResult] = []
    audience_path = ROOT_DIR / "src" / "edutech_vision" / "modes" / "audience.py"
    audience_code = audience_path.read_text(encoding="utf-8")
    demo_code = (ROOT_DIR / "src" / "edutech_vision" / "modes" / "demo.py").read_text(encoding="utf-8")
    launcher_code = (ROOT_DIR / "src" / "edutech_vision" / "launcher.py").read_text(encoding="utf-8")
    report_code = (ROOT_DIR / "scripts" / "generate_session_report.py").read_text(encoding="utf-8")

    checks.append(
        result(
            section,
            "Modo Plateia registra telemetria suficiente para graficos.",
            all(field in audience_code for field in ["faces_mean", "attentive_mean", "engagement_percent", "session_note"]),
            "campos de plateia",
        )
    )
    checks.append(
        result(
            section,
            "Painel de plateia usa barras, timeline e insight explicativo.",
            all(token in audience_code for token in ["bars=", "events=", "insight="]),
            "compose_dashboard",
        )
    )
    checks.append(result(section, "Modo Demo tem chamada visual positiva.", "FLUXO COMPLETO EM TEMPO REAL" in demo_code, "banner demo"))
    checks.append(result(section, "Control Center oferece navegacao visual.", "class LauncherApp" in launcher_code and "TNotebook" in launcher_code, "launcher.py"))
    checks.append(result(section, "Control Center permite escolher detector.", "self.detector" in launcher_code and "benchmark_smoke" in launcher_code, "launcher.py"))
    checks.append(
        result(
            section,
            "Relatorio gera Markdown, HTML, PDF e graficos.",
            all(token in report_code for token in ["write_pdf_report", "markdown_to_simple_html", "generate_charts"]),
            "generate_session_report.py",
        )
    )
    checks.append(manual(section, "Explicar que o sistema gera indicadores visuais aproximados, nao diagnostico medico.", "frase consta em docs/DEFESA_ORAL.md"))
    return checks


def check_performance_defaults() -> list[CheckResult]:
    section = "9. Performance e estabilidade"
    config = (ROOT_DIR / "src" / "edutech_vision" / "config.py").read_text(encoding="utf-8")
    checks = [
        result(section, "Comecar com resolucao `960x540`.", "width: int = 960" in config and "height: int = 540" in config, "AppConfig defaults"),
        manual(section, "Se FPS ficar abaixo de 20, reduzir para `640x480`.", "comando documentado na checklist"),
        manual(section, "No Modo Plateia, reduzir `--max-faces` se a maquina estiver lenta.", "argumento existe no CLI"),
        manual(section, "Evitar deixar outros apps pesados abertos.", "operacional"),
        manual(section, "Rodar `test_fps.py` no mesmo notebook da apresentacao.", "requer hardware final"),
    ]
    detector = run(
        [
            sys.executable,
            "-c",
            "from edutech_vision.core.landmarks import FaceLandmarkDetector; "
            "d=FaceLandmarkDetector(max_faces=1, detector_profile='enhanced'); d.close(); print('ok')",
        ]
    )
    checks.append(result(section, "Verificar se o detector `enhanced` inicializa antes da banca chegar.", detector.ok, first_line(detector.stdout, detector.stderr)))
    checks.append(manual(section, "Manter terminal com comandos recentes no historico.", "operacional"))
    return checks


def check_code_explanation_files() -> list[CheckResult]:
    section = "11. Arquivos de codigo para explicar se sorteados"
    return [result(section, f"`{path}` existe.", (ROOT_DIR / path).exists(), path) for path in CODE_EXPLANATION_FILES]


def manual_demo_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    section = "5. Fluxo da demo ao vivo"
    for item in [
        "Abrir terminal na raiz do projeto.",
        "Rodar `scripts/doctor.py`.",
        "Rodar Modo Individual.",
        "Aguardar calibracao inicial de 5 segundos.",
        "Mostrar face detectada e landmarks/pontos principais.",
        "Mostrar EAR mudando ao fechar os olhos.",
        "Mostrar MAR mudando ao abrir a boca.",
        "Mostrar yaw/pitch mudando ao virar/inclinar a cabeca.",
        "Demonstrar alerta de fadiga sustentada sem depender de piscada curta.",
        "Demonstrar alerta de postura com inclinacao sustentada.",
        "Demonstrar alerta de desatencao virando o rosto por tempo suficiente.",
        "Fechar com `Q` ou `Esc`, sem matar processo pelo terminal.",
        "Rodar Modo Plateia.",
        "Mostrar contagem de faces, barras e percentual agregado.",
        "Mostrar timeline de eventos no painel lateral.",
        "Mostrar que o relatorio PDF/HTML nasce dos logs CSV.",
        "Se a webcam falhar, rodar `.\\.venv\\Scripts\\python.exe -m edutech_vision --mode demo --showcase` e apresentar como demo sintetica.",
        "Abrir `results/session_report.pdf` e explicar os graficos principais.",
    ]:
        checks.append(manual(section, item, "acao de demonstracao ao vivo"))

    for item in [
        '"EAR e MAR sao razoes geometricas, nao classificadores treinados."',
        '"A pose de cabeca vem de `solvePnP`, usando pontos 2D da imagem e um modelo 3D generico."',
        '"A suavizacao temporal reduz falsos positivos de eventos pontuais."',
        '"Alertas so disparam quando a condicao e sustentada."',
        '"Modo Plateia converte varias faces em percentual de engajamento por janela."',
        '"Os logs sao evidencias numericas para reproducibilidade e artigo."',
        '"O sistema e uma ferramenta academica de PDI, nao diagnostico clinico."',
    ]:
        checks.append(manual("12. Frases tecnicas alinhadas", item, "ensaiar oralmente"))

    for item in [
        "Nao prometer deteccao perfeita de atencao.",
        "Nao dizer que identifica aluno desatento individualmente no Modo Plateia.",
        "Nao usar `labels_demo.csv` como resultado cientifico final.",
        "Nao encerrar processo de forma abrupta se a webcam falhar; mostrar a reconexao.",
        "Nao deixar a explicacao virar defesa de limitacoes; voltar para pipeline, resultados e demo.",
    ]:
        checks.append(manual("13. Nao fazer durante a apresentacao", item, "regra operacional"))

    for item in [
        "Fechar app com `Q` ou `Esc`.",
        "Conferir logs gerados em `results/`.",
        "Rodar `scripts/summarize_results.py` se houve nova coleta.",
        "Criar tag final se a versao apresentada mudou.",
    ]:
        checks.append(manual("14. Encerramento apos a apresentacao", item, "executar apos demo/coleta"))

    return checks


def check_requirements_pinned() -> CheckResult:
    path = ROOT_DIR / "requirements.txt"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    unpinned = [line for line in lines if "==" not in line]
    return result("2. Ambiente Python", "Dependencias estao nas versoes pinadas em `requirements.txt`.", not unpinned, "todas pinadas" if not unpinned else "; ".join(unpinned))


def check_requirements_match_pyproject() -> CheckResult:
    requirements = set(read_requirement_lines(ROOT_DIR / "requirements.txt"))
    dev_requirements = set(read_requirement_lines(ROOT_DIR / "requirements-dev.txt"))
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    project_requirements = set(project.get("dependencies", []))
    project_dev = set(project.get("optional-dependencies", {}).get("dev", []))

    mismatches = []
    if requirements != project_requirements:
        mismatches.append("runtime")
    if dev_requirements != project_dev:
        mismatches.append("dev")
    ok = not mismatches
    detail = "requirements.txt, requirements-dev.txt e pyproject.toml alinhados" if ok else "divergencia: " + ", ".join(mismatches)
    return result("2. Ambiente Python", "Dependencias de requirements e pyproject estao sincronizadas.", ok, detail)


def read_requirement_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def check_installed_editable() -> CheckResult:
    package = "secomp-edutech-vision"
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return CheckResult("2. Ambiente Python", "Pacote local esta instalado em modo editavel.", "FAIL", f"{package} nao instalado")

    direct_url = distribution.read_text("direct_url.json") or ""
    editable = '"editable": true' in direct_url.lower()
    return result("2. Ambiente Python", "Pacote local esta instalado em modo editavel.", editable, "editable install" if editable else "instalado, mas nao editavel")


def can_import_launcher() -> bool:
    launcher = run([sys.executable, "-c", "import edutech_vision.launcher; print('ok')"])
    return launcher.ok


def module_available(name: str) -> bool:
    probe = run([sys.executable, "-c", f"import {name}; print('ok')"])
    return probe.ok


def check_results_are_numeric_or_aggregate() -> CheckResult:
    bad_files: list[str] = []
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and path.name not in ALLOWED_RESULTS_MEDIA:
                bad_files.append(str(path.relative_to(ROOT_DIR)))
    return result(
        "1. Estado do repositorio",
        "`results/` contem somente evidencias numericas e agregadas adequadas.",
        not bad_files,
        "ok" if not bad_files else "; ".join(bad_files),
    )


def check_lighting_csv() -> CheckResult:
    path = ROOT_DIR / "results" / "lighting_evaluation.csv"
    rows = read_rows(path)
    stages = {row.get("stage") for row in rows if row.get("stage")}
    return warn_ok("7. Protocolos ao vivo", "CSV de iluminacao registra os tres estagios.", {"baixa", "media", "alta"}.issubset(stages), f"estagios: {sorted(stages)}")


def check_occlusion_csv() -> CheckResult:
    path = ROOT_DIR / "results" / "occlusion_recovery.csv"
    rows = read_rows(path)
    phases = {row.get("phase") for row in rows if row.get("phase")}
    return warn_ok("7. Protocolos ao vivo", "CSV de oclusao registra baseline, oclusao e recuperacao.", {"baseline", "oclusao", "recuperacao"}.issubset(phases), f"fases: {sorted(phases)}")


def check_fps_csv() -> CheckResult:
    path = ROOT_DIR / "results" / "fps_log.csv"
    rows = read_rows(path)
    values = [float(row.get("fps_current") or 0) for row in rows if row.get("fps_current")]
    if not values:
        return CheckResult("7. Protocolos ao vivo", "FPS medio >= 20 e estatisticas reportaveis.", "WARN", "fps_log.csv ausente ou vazio")
    mean = sum(values) / len(values)
    return warn_ok("7. Protocolos ao vivo", "FPS medio >= 20 e estatisticas reportaveis.", mean >= 20.0, f"media {mean:.2f}; n={len(values)}")


def check_real_labels() -> CheckResult:
    path = ROOT_DIR / "assets" / "samples" / "labels_real.csv"
    rows = count_csv_rows(path)
    if rows >= 50:
        return CheckResult("7. Protocolos ao vivo", "Matriz de confusao real esta preenchida.", "OK", f"{rows} amostras reais")
    return manual(
        "7. Protocolos ao vivo",
        "Matriz de confusao real e opcional; a demonstracao usa `labels_demo.csv` quando `labels_real.csv` esta vazio.",
        f"{rows} amostras reais",
    )


def check_confusion_metrics() -> CheckResult:
    path = ROOT_DIR / "results" / "confusion_metrics.csv"
    rows = read_rows(path)
    if not rows:
        return CheckResult("7. Protocolos ao vivo", "Matriz de confusao tem 50+ amostras e acuracia >= 70%.", "WARN", "confusion_metrics.csv ausente")
    row = rows[-1]
    samples = int(float(row.get("samples") or 0))
    accuracy = float(row.get("accuracy") or 0)
    return warn_ok("7. Protocolos ao vivo", "Matriz de confusao tem 50+ amostras e acuracia >= 70%.", samples >= 50 and accuracy >= 0.70, f"n={samples}; acc={accuracy:.3f}")


def check_failover_csv() -> CheckResult:
    path = ROOT_DIR / "results" / "failover_log.csv"
    rows = read_rows(path)
    read_values = [int(float(row.get("read_ok") or 0)) for row in rows]
    saw_failure = 0 in read_values
    recovered = False
    if saw_failure:
        first_failure = read_values.index(0)
        recovered = any(value == 1 for value in read_values[first_failure + 1 :])
    return warn_ok("7. Protocolos ao vivo", "Failover registra falha e retomada sem crash.", saw_failure and recovered, "recuperado" if recovered else "pendente")


def tracked_media_files() -> list[str]:
    files = run(["git", "ls-files"]).stdout.splitlines()
    return [path for path in files if Path(path).suffix.lower() in MEDIA_EXTENSIONS]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def count_csv_rows(path: Path) -> int:
    return len(read_rows(path))


def write_report(path: Path, checks: list[CheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relatorio de Preflight da Apresentacao",
        "",
        "Status possiveis: `OK`, `WARN`, `FAIL`, `MANUAL`.",
        "",
    ]
    current_section = ""
    for check in checks:
        if check.section != current_section:
            current_section = check.section
            lines.extend(["", f"## {current_section}", ""])
            lines.append("| Status | Item | Detalhe |")
            lines.append("| --- | --- | --- |")
        detail = check.detail.replace("|", "\\|").replace("\n", " ")
        item = check.item.replace("|", "\\|")
        lines.append(f"| {check.status} | {item} | {detail} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(checks: list[CheckResult], report_path: Path) -> None:
    counts = {status: sum(1 for check in checks if check.status == status) for status in ["OK", "WARN", "FAIL", "MANUAL"]}
    print(f"OK={counts['OK']} WARN={counts['WARN']} FAIL={counts['FAIL']} MANUAL={counts['MANUAL']}")
    print(f"Relatorio: {report_path}")
    for check in checks:
        if check.status in {"FAIL", "WARN"}:
            print(f"[{check.status}] {check.section} - {check.item}: {check.detail}")


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str
    stderr: str


def run(command: list[str]) -> CommandResult:
    completed = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True, check=False)
    return CommandResult(completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip())


def result(section: str, item: str, ok: bool, detail: str) -> CheckResult:
    return CheckResult(section, item, "OK" if ok else "FAIL", detail)


def warn_ok(section: str, item: str, ok: bool, detail: str) -> CheckResult:
    return CheckResult(section, item, "OK" if ok else "WARN", detail)


def manual(section: str, item: str, detail: str) -> CheckResult:
    return CheckResult(section, item, "MANUAL", detail)


def first_line(stdout: str, stderr: str) -> str:
    text = stdout or stderr
    if not text:
        return ""
    return text.splitlines()[0]


def last_relevant_line(stdout: str, stderr: str) -> str:
    lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if "[OK]" in line or "[ERRO]" in line:
            return line
    return lines[-1]


if __name__ == "__main__":
    main()
