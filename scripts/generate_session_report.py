from __future__ import annotations

import argparse
import csv
import html
import math
import random
import statistics
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
CHART_FILENAMES = {
    "summary": "session_summary_dashboard.png",
    "individual": "session_individual_timeline.png",
    "audience": "session_audience_timeline.png",
    "alerts": "session_alerts.png",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera relatorio visual de sessao.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--charts-dir", type=Path)
    parser.add_argument(
        "--demo-synthetic",
        action="store_true",
        help="Gera um relatorio de simulacao com dados sinteticos.",
    )
    parser.add_argument("--demo-output-dir", type=Path, default=RESULTS_DIR / "demo_showcase")
    args = parser.parse_args()

    results_dir = args.demo_output_dir if args.demo_synthetic else args.results_dir
    if args.demo_synthetic:
        create_synthetic_demo_results(results_dir)

    output_md = args.output_md or results_dir / ("session_report_demo.md" if args.demo_synthetic else "session_report.md")
    output_html = args.output_html or results_dir / ("session_report_demo.html" if args.demo_synthetic else "session_report.html")
    output_pdf = args.output_pdf or results_dir / ("session_report_demo.pdf" if args.demo_synthetic else "session_report.pdf")
    charts_dir = args.charts_dir or results_dir / "report_charts"

    chart_paths = generate_charts(results_dir, charts_dir, synthetic=args.demo_synthetic)
    report = build_report(results_dir, synthetic=args.demo_synthetic, chart_paths=chart_paths, output_dir=output_md.parent)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    output_html.write_text(markdown_to_simple_html(report), encoding="utf-8")
    write_pdf_report(report, chart_paths, output_pdf, synthetic=args.demo_synthetic)

    print(f"Relatorio Markdown: {output_md}")
    print(f"Relatorio HTML: {output_html}")
    print(f"Relatorio PDF: {output_pdf}")
    print(f"Graficos agregados: {charts_dir}")


def build_report(
    results_dir: Path,
    *,
    synthetic: bool = False,
    chart_paths: dict[str, Path] | None = None,
    output_dir: Path | None = None,
) -> str:
    chart_paths = chart_paths or {}
    output_dir = output_dir or results_dir
    title = "Relatorio de Sessao - EduTech Vision"
    lines = [
        f"# {title}",
        "",
        "Este relatorio transforma os CSVs da sessao em graficos, estatisticas e linha do tempo para apresentacao.",
        "",
    ]
    if synthetic:
        lines.extend(
            [
                "> DEMO SINTETICA: dados gerados para exibir o formato visual do relatorio.",
                "> Para a entrega, gere novamente usando os CSVs reais dos protocolos.",
                "",
            ]
        )
    lines.extend(visual_artifacts_section(chart_paths, output_dir))
    lines.extend(individual_section(results_dir / "individual_session.csv"))
    lines.extend(audience_section(results_dir / "audience_engagement.csv"))
    lines.extend(events_section(results_dir / "presentation_events.csv"))
    lines.extend(protocol_section(results_dir / "protocol_summary.md"))
    lines.extend(scope_section())
    return "\n".join(lines) + "\n"


def visual_artifacts_section(chart_paths: dict[str, Path], output_dir: Path) -> list[str]:
    lines = ["## Artefatos Visuais", ""]
    if not chart_paths:
        return lines + ["Graficos agregados ainda nao gerados.", ""]

    labels = {
        "summary": "Resumo executivo",
        "individual": "Linha temporal do modo individual",
        "audience": "Linha temporal do modo plateia",
        "alerts": "Distribuicao de eventos e alertas",
    }
    for key in ("summary", "individual", "audience", "alerts"):
        path = chart_paths.get(key)
        if path is None:
            continue
        rel = relative_markdown_path(path, output_dir)
        lines.extend([f"![{labels[key]}]({rel})", ""])
    return lines


def individual_section(path: Path) -> list[str]:
    rows = read_rows(path)
    lines = ["## Modo Individual", ""]
    if not rows:
        return lines + ["Sessao individual ainda nao registrada.", ""]

    stats = individual_stats(rows)
    lines.extend(
        [
            f"- Frames/logs analisados: {stats['total']:.0f}",
            f"- Taxa de face detectada: {stats['face_rate']:.1f}%",
            f"- FPS medio observado: {stats['fps_mean']:.2f}",
            f"- EAR medio: {stats['ear_mean']:.3f}",
            f"- MAR medio: {stats['mar_mean']:.3f}",
            f"- Alertas por frame: {format_counter(stats['alert_counts'])}",
            "",
        ]
    )
    return lines


def audience_section(path: Path) -> list[str]:
    rows = read_rows(path)
    lines = ["## Modo Plateia", ""]
    if not rows:
        return lines + ["Sessao de plateia ainda nao registrada.", ""]

    stats = audience_stats(rows)
    lines.extend(
        [
            f"- Janelas agregadas registradas: {stats['windows']:.0f}",
            f"- Engajamento medio: {stats['engagement_mean']:.1f}%",
            f"- Engajamento minimo/maximo: {stats['engagement_min']:.1f}% / {stats['engagement_max']:.1f}%",
            f"- Media de faces por janela: {stats['faces_mean']:.2f}",
            f"- Media de faces atentas por janela: {stats['attentive_mean']:.2f}",
            f"- FPS medio: {stats['fps_mean']:.2f}",
            "- Leitura: o percentual mostra a proporcao estimada de faces voltadas ao palco em cada janela.",
            "",
        ]
    )
    return lines


def events_section(path: Path) -> list[str]:
    rows = read_rows(path)
    lines = ["## Linha do Tempo de Eventos", ""]
    if not rows:
        return lines + ["Nenhum evento discreto registrado ainda.", ""]

    counts = Counter(row.get("event", "indefinido") for row in rows)
    lines.append(f"- Eventos registrados: {len(rows)}")
    lines.append(f"- Distribuicao: {format_counter(dict(counts))}")
    lines.append("")
    lines.append("| Tempo | Modo | Evento | Estado | Detalhe |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows[-12:]:
        lines.append(
            f"| {row.get('seconds', '')} | {row.get('mode', '')} | {row.get('event', '')} | "
            f"{row.get('state', '')} | {row.get('detail', '')} |"
        )
    lines.append("")
    return lines


def protocol_section(path: Path) -> list[str]:
    if not path.exists():
        return ["## Protocolos", "", "Resumo dos protocolos ainda nao gerado.", ""]
    content = path.read_text(encoding="utf-8").strip()
    return ["## Protocolos", "", content, ""]


def scope_section() -> list[str]:
    return [
        "## Escopo Tecnico",
        "",
        "- O sistema deve ser apresentado como ferramenta academica de PDI, nao diagnostico clinico.",
        "- EAR, MAR e pose de cabeca sao metricas geometricas sensiveis a iluminacao, distancia e calibracao.",
        "- A leitura de plateia e uma estimativa temporal de engajamento, nao uma avaliacao individual de aprendizagem.",
        "- Os graficos ajudam a explicar o comportamento do pipeline durante a demonstracao.",
        "",
    ]


def generate_charts(results_dir: Path, charts_dir: Path, *, synthetic: bool = False) -> dict[str, Path]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    individual_rows = read_rows(results_dir / "individual_session.csv")
    audience_rows = read_rows(results_dir / "audience_engagement.csv")
    event_rows = read_rows(results_dir / "presentation_events.csv")

    chart_paths = {
        key: charts_dir / filename for key, filename in CHART_FILENAMES.items()
    }
    plot_summary_dashboard(individual_rows, audience_rows, event_rows, chart_paths["summary"], synthetic=synthetic)
    plot_individual_timeline(individual_rows, chart_paths["individual"], synthetic=synthetic)
    plot_audience_timeline(audience_rows, chart_paths["audience"], synthetic=synthetic)
    plot_alert_distribution(individual_rows, event_rows, chart_paths["alerts"], synthetic=synthetic)
    return chart_paths


def plot_summary_dashboard(
    individual_rows: list[dict[str, str]],
    audience_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    output: Path,
    *,
    synthetic: bool,
) -> None:
    ind = individual_stats(individual_rows)
    aud = audience_stats(audience_rows)
    alert_total = sum(ind["alert_counts"].values())
    event_total = len(event_rows)
    labels = ["Face %", "FPS ind.", "Engaj. %", "Faces", "Alertas", "Eventos"]
    values = [
        ind["face_rate"],
        ind["fps_mean"],
        aud["engagement_mean"],
        aud["faces_mean"],
        float(alert_total),
        float(event_total),
    ]
    colors = ["#2d9cdb", "#27ae60", "#f2c94c", "#9b51e0", "#eb5757", "#56cc9d"]

    with report_style():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(labels, values, color=colors)
        ax.set_title("Resumo executivo da sessao", fontweight="bold")
        ax.set_ylabel("Valor agregado")
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            ax.text(index, value + max(values + [1]) * 0.02, f"{value:.1f}", ha="center", fontsize=9)
        add_watermark(ax, synthetic)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def plot_individual_timeline(rows: list[dict[str, str]], output: Path, *, synthetic: bool) -> None:
    seconds = floats(rows, "seconds")
    ear = floats(rows, "ear")
    mar = floats(rows, "mar")
    with report_style():
        fig, ax = plt.subplots(figsize=(12, 5))
        if seconds and ear:
            ax.plot(seconds[: len(ear)], ear, label="EAR", color="#2d9cdb", linewidth=1.8)
            if mar:
                ax.plot(seconds[: len(mar)], mar, label="MAR", color="#f2994a", linewidth=1.8)
            ax.set_ylabel("Razao geometrica")
            ax.set_xlabel("Tempo (s)")
            ax.legend()
            ax.grid(alpha=0.25)
        else:
            draw_placeholder(ax, "Aguardando CSV real do modo individual")
        ax.set_title("Modo Individual: EAR/MAR ao longo do tempo", fontweight="bold")
        add_watermark(ax, synthetic)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def plot_audience_timeline(rows: list[dict[str, str]], output: Path, *, synthetic: bool) -> None:
    seconds = floats(rows, "seconds")
    engagement = floats(rows, "engagement_percent")
    faces = floats(rows, "faces_mean")
    with report_style():
        fig, ax1 = plt.subplots(figsize=(12, 5))
        if seconds and engagement:
            ax1.plot(seconds[: len(engagement)], engagement, color="#27ae60", linewidth=2.0, label="Engajamento 10s")
            ax1.fill_between(seconds[: len(engagement)], engagement, color="#27ae60", alpha=0.12)
            ax1.set_ylabel("Engajamento (%)")
            ax1.set_ylim(0, 100)
            ax2 = ax1.twinx()
            ax2.plot(seconds[: len(faces)], faces, color="#9b51e0", linewidth=1.5, label="Faces medias")
            ax2.set_ylabel("Faces medias")
            ax1.set_xlabel("Tempo (s)")
            ax1.grid(alpha=0.25)
        else:
            draw_placeholder(ax1, "Aguardando CSV real do modo plateia")
        ax1.set_title("Modo Plateia: engajamento agregado por janela", fontweight="bold")
        add_watermark(ax1, synthetic)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def plot_alert_distribution(
    individual_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    output: Path,
    *,
    synthetic: bool,
) -> None:
    alert_counts = individual_stats(individual_rows)["alert_counts"]
    event_counts = Counter(row.get("event", "indefinido") for row in event_rows)
    combined = {f"alerta:{key}": value for key, value in alert_counts.items() if value}
    combined.update({f"evento:{key}": value for key, value in event_counts.items() if value})

    with report_style():
        fig, ax = plt.subplots(figsize=(12, 5))
        if combined:
            labels = list(combined)
            values = [combined[label] for label in labels]
            ax.barh(labels, values, color="#eb5757")
            ax.set_xlabel("Ocorrencias")
            ax.grid(axis="x", alpha=0.25)
        else:
            draw_placeholder(ax, "Aguardando eventos ou alertas registrados")
        ax.set_title("Eventos e alertas discretos", fontweight="bold")
        add_watermark(ax, synthetic)
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)


def write_pdf_report(report_markdown: str, chart_paths: dict[str, Path], output: Path, *, synthetic: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        title = "Relatorio de Sessao - EduTech Vision"
        if synthetic:
            title += " | SIMULACAO"
        fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color="#123f55")
        body = markdown_to_plaintext(report_markdown)
        wrapped = []
        for line in body.splitlines():
            wrapped.extend(textwrap.wrap(line, width=92) or [""])
        y = 0.89
        for line in wrapped[:44]:
            fig.text(0.08, y, line, fontsize=9.5, color="#17202a")
            y -= 0.018
        if synthetic:
            fig.text(
                0.08,
                0.08,
                "DEMO SINTETICA: gere novamente com CSVs reais para a entrega.",
                fontsize=10,
                fontweight="bold",
                color="#b00020",
            )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for key in ("summary", "individual", "audience", "alerts"):
            path = chart_paths.get(key)
            if path is None or not path.exists():
                continue
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            image = plt.imread(path)
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def create_synthetic_demo_results(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    write_synthetic_individual(output_dir / "individual_session.csv", rng)
    write_synthetic_audience(output_dir / "audience_engagement.csv", rng)
    write_synthetic_events(output_dir / "presentation_events.csv")
    (output_dir / "protocol_summary.md").write_text(
        "# Resumo dos Protocolos - SIMULACAO\n\n"
        "Estes dados sao sinteticos e existem para demonstrar o formato do relatorio.\n"
        "Para a entrega, gere novamente com os CSVs reais coletados pela equipe.\n",
        encoding="utf-8",
    )


def write_synthetic_individual(path: Path, rng: random.Random) -> None:
    fields = [
        "timestamp",
        "seconds",
        "face_detected",
        "ear",
        "mar",
        "yaw",
        "pitch",
        "roll",
        "eye_closed_alert",
        "yawn_alert",
        "posture_alert",
        "distraction_alert",
        "fatigue_alert",
        "fps_current",
        "fps_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        fps_window: list[float] = []
        for index in range(240):
            seconds = index * 0.5
            eye_event = 54 <= index <= 60 or 166 <= index <= 172
            yawn_event = 112 <= index <= 118
            posture_event = 150 <= index <= 176
            distraction_event = 78 <= index <= 96
            fps = 27.5 + math.sin(seconds / 7.0) * 1.3 + rng.uniform(-0.5, 0.5)
            fps_window.append(fps)
            fps_window = fps_window[-30:]
            ear = 0.29 + math.sin(seconds / 9.0) * 0.012 + rng.uniform(-0.008, 0.008)
            mar = 0.075 + math.sin(seconds / 11.0) * 0.006 + rng.uniform(-0.004, 0.004)
            if eye_event:
                ear = 0.13 + rng.uniform(-0.01, 0.006)
            if yawn_event:
                mar = 0.29 + rng.uniform(-0.015, 0.015)
            yaw = 4.0 * math.sin(seconds / 5.5)
            if distraction_event:
                yaw = 28.0 + rng.uniform(-4.0, 4.0)
            pitch = 3.0 * math.sin(seconds / 8.0)
            if posture_event:
                pitch = 19.0 + rng.uniform(-2.0, 2.0)
            writer.writerow(
                {
                    "timestamp": "SIMULACAO",
                    "seconds": f"{seconds:.3f}",
                    "face_detected": "1",
                    "ear": f"{ear:.5f}",
                    "mar": f"{mar:.5f}",
                    "yaw": f"{yaw:.3f}",
                    "pitch": f"{pitch:.3f}",
                    "roll": f"{1.5 * math.sin(seconds / 10.0):.3f}",
                    "eye_closed_alert": int(eye_event),
                    "yawn_alert": int(yawn_event),
                    "posture_alert": int(posture_event),
                    "distraction_alert": int(distraction_event),
                    "fatigue_alert": int(eye_event or yawn_event),
                    "fps_current": f"{fps:.3f}",
                    "fps_mean": f"{statistics.fmean(fps_window):.3f}",
                }
            )


def write_synthetic_audience(path: Path, rng: random.Random) -> None:
    fields = [
        "timestamp",
        "seconds",
        "window_seconds",
        "faces_mean",
        "attentive_mean",
        "engagement_percent",
        "fps_mean",
        "session_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(1, 25):
            seconds = index * 10.0
            faces = 8.5 + 2.5 * math.sin(seconds / 35.0) + rng.uniform(-0.6, 0.6)
            engagement = 66.0 + 14.0 * math.sin(seconds / 42.0) - 8.0 * math.exp(-((seconds - 145.0) ** 2) / 900.0)
            engagement = max(35.0, min(90.0, engagement + rng.uniform(-3.0, 3.0)))
            attentive = faces * engagement / 100.0
            writer.writerow(
                {
                    "timestamp": "SIMULACAO",
                    "seconds": f"{seconds:.3f}",
                    "window_seconds": "10.0",
                    "faces_mean": f"{faces:.3f}",
                    "attentive_mean": f"{attentive:.3f}",
                    "engagement_percent": f"{engagement:.3f}",
                    "fps_mean": f"{23.0 + math.sin(seconds / 30.0):.3f}",
                    "session_note": "SIMULACAO; janela 10s pronta para graficos",
                }
            )


def write_synthetic_events(path: Path) -> None:
    fields = ["timestamp", "seconds", "mode", "event", "state", "detail"]
    events = [
        (5.0, "individual", "Rastreamento", "on", "face sintetica presente"),
        (29.0, "individual", "Desatencao", "on", "yaw simulado acima do limiar"),
        (48.0, "individual", "Desatencao", "off", "normalizado"),
        (58.0, "individual", "Fadiga", "on", "EAR simulado baixo"),
        (64.0, "individual", "Fadiga", "off", "normalizado"),
        (115.0, "individual", "Bocejo", "on", "MAR simulado alto"),
        (121.0, "individual", "Bocejo", "off", "normalizado"),
        (150.0, "individual", "Postura", "on", "pitch simulado sustentado"),
        (178.0, "individual", "Postura", "off", "normalizado"),
        (190.0, "plateia", "Janela 10s", "sample", "engajamento sintetico 72%"),
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for seconds, mode, event, state, detail in events:
            writer.writerow(
                {
                    "timestamp": "SIMULACAO",
                    "seconds": f"{seconds:.3f}",
                    "mode": mode,
                    "event": event,
                    "state": state,
                    "detail": detail,
                }
            )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def floats(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field) or 0.0))
        except ValueError:
            continue
    return values


def count_true(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if row.get(field) in {"1", "true", "True"})


def individual_stats(rows: list[dict[str, str]]) -> dict[str, float | dict[str, int]]:
    face_rows = [row for row in rows if row.get("face_detected") == "1"]
    total = len(rows)
    alert_counts = {
        "fadiga": count_true(rows, "fatigue_alert"),
        "bocejo": count_true(rows, "yawn_alert"),
        "postura": count_true(rows, "posture_alert"),
        "desatencao": count_true(rows, "distraction_alert"),
    }
    return {
        "total": float(total),
        "face_rate": 100.0 * len(face_rows) / total if total else 0.0,
        "fps_mean": mean(floats(rows, "fps_current")),
        "ear_mean": mean(floats(face_rows, "ear")),
        "mar_mean": mean(floats(face_rows, "mar")),
        "alert_counts": alert_counts,
    }


def audience_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    engagement = floats(rows, "engagement_percent")
    faces = floats(rows, "faces_mean")
    attentive = floats(rows, "attentive_mean")
    fps = floats(rows, "fps_mean")
    return {
        "windows": float(len(rows)),
        "engagement_mean": mean(engagement),
        "engagement_min": minimum(engagement),
        "engagement_max": maximum(engagement),
        "faces_mean": mean(faces),
        "attentive_mean": mean(attentive),
        "fps_mean": mean(fps),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def minimum(values: list[float]) -> float:
    return min(values) if values else 0.0


def maximum(values: list[float]) -> float:
    return max(values) if values else 0.0


def format_counter(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) if values else "sem eventos"


def report_style():
    return plt.rc_context(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "axes.edgecolor": "#d8e1e8",
            "axes.labelcolor": "#17202a",
            "xtick.color": "#34495e",
            "ytick.color": "#34495e",
            "text.color": "#17202a",
            "grid.color": "#cfd8df",
            "font.family": "DejaVu Sans",
        }
    )


def draw_placeholder(ax: plt.Axes, message: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="#607080", transform=ax.transAxes)


def add_watermark(ax: plt.Axes, synthetic: bool) -> None:
    if not synthetic:
        return
    ax.text(
        0.99,
        0.04,
        "SIMULACAO - nao e evidencia",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=11,
        color="#b00020",
        fontweight="bold",
        alpha=0.9,
    )


def relative_markdown_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def markdown_to_plaintext(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("!["):
            continue
        if line.startswith("#"):
            lines.append(line.lstrip("# ").strip())
        elif line.startswith(">"):
            lines.append(line.lstrip("> ").strip())
        elif line.startswith("|"):
            continue
        else:
            lines.append(line)
    return "\n".join(lines)


def markdown_to_simple_html(markdown: str) -> str:
    body_lines: list[str] = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt, src = parse_markdown_image(line)
            body_lines.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}">'
                f"<figcaption>{html.escape(alt)}</figcaption></figure>"
            )
        elif line.startswith("> "):
            body_lines.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.startswith("- "):
            body_lines.append(f"<p>{html.escape(line)}</p>")
        elif line.startswith("|"):
            if set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            if not in_table:
                body_lines.append("<table>")
                in_table = True
            body_lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        else:
            if in_table:
                body_lines.append("</table>")
                in_table = False
            if line.strip():
                body_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body_lines.append("</table>")

    return """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Relatorio de Sessao - EduTech Vision</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #17202a; background: #fbfcfd; }
    h1 { color: #123f55; }
    h2 { color: #1f6f8b; border-bottom: 1px solid #d8e1e8; padding-bottom: 6px; margin-top: 32px; }
    p { line-height: 1.45; }
    blockquote { border-left: 5px solid #b00020; margin: 16px 0; padding: 10px 14px; background: #fff3f3; font-weight: 700; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; background: white; }
    td { border: 1px solid #d8e1e8; padding: 8px; }
    tr:first-child { font-weight: 700; background: #eef6f8; }
    figure { margin: 18px 0 28px; padding: 14px; background: white; border: 1px solid #d8e1e8; }
    img { max-width: 100%; display: block; }
    figcaption { color: #52616b; font-size: 0.92rem; margin-top: 8px; }
  </style>
</head>
<body>
""" + "\n".join(body_lines) + "\n</body>\n</html>\n"


def parse_markdown_image(line: str) -> tuple[str, str]:
    alt_end = line.find("]")
    src_start = line.find("(", alt_end)
    alt = line[2:alt_end]
    src = line[src_start + 1 : -1]
    return alt, src


if __name__ == "__main__":
    main()
