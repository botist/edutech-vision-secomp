from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProtocolSummary:
    protocol: str
    evidence: str
    metric: str
    target: str
    status: str
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera resumo dos resultados dos cinco protocolos.")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--output-md", type=Path, default=ROOT_DIR / "results" / "protocol_summary.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "results" / "protocol_summary.csv")
    args = parser.parse_args()

    summaries = summarize_all(args.results_dir)
    write_markdown(args.output_md, summaries)
    write_csv(args.output_csv, summaries)

    print(f"Resumo Markdown: {args.output_md}")
    print(f"Resumo CSV: {args.output_csv}")


def summarize_all(results_dir: Path) -> list[ProtocolSummary]:
    return [
        summarize_lighting(results_dir / "lighting_evaluation.csv"),
        summarize_occlusion(results_dir / "occlusion_recovery.csv"),
        summarize_fps(results_dir / "fps_log.csv"),
        summarize_confusion(results_dir / "confusion_metrics.csv"),
        summarize_failover(results_dir / "failover_log.csv"),
    ]


def summarize_lighting(path: Path) -> ProtocolSummary:
    rows = read_rows(path)
    if not rows:
        return pending("Robustez luminosa", path, "Taxa de deteccao por estagio", "3 niveis, degradacao <= 30%")

    by_stage: dict[str, list[int]] = {}
    for row in rows:
        stage = row.get("stage", "indefinido")
        by_stage.setdefault(stage, []).append(int(float(row.get("faces_detected") or 0) > 0))

    rates = {stage: 100.0 * sum(values) / len(values) for stage, values in by_stage.items() if values}
    if len(rates) < 3:
        status = "atenção"
        notes = "menos de 3 niveis de iluminacao registrados"
    else:
        values = list(rates.values())
        degradation = max(values) - min(values)
        status = "ok" if degradation <= 30.0 else "atenção"
        notes = f"degradacao estimada por deteccao: {degradation:.1f} p.p."

    metric = "; ".join(f"{stage}: {rate:.1f}%" for stage, rate in sorted(rates.items()))
    return ProtocolSummary("Robustez luminosa", str(path), metric, "3 niveis, degradacao <= 30%", status, notes)


def summarize_occlusion(path: Path) -> ProtocolSummary:
    rows = read_rows(path)
    if not rows:
        return pending("Resiliencia a oclusao", path, "Tempo ate redeteccao", "retoma <= 2 s")

    recovery_start = None
    recovered = None
    for row in rows:
        seconds = parse_float(row.get("seconds"))
        if row.get("phase") == "recuperacao" and recovery_start is None:
            recovery_start = seconds
        if row.get("phase") == "recuperacao" and int(float(row.get("faces_detected") or 0)) > 0:
            recovered = seconds
            break

    if recovery_start is None or recovered is None:
        return ProtocolSummary("Resiliencia a oclusao", str(path), "nao recuperou no log", "retoma <= 2 s", "atenção", "reexecutar protocolo")

    delta = recovered - recovery_start
    status = "ok" if delta <= 2.0 else "atenção"
    return ProtocolSummary("Resiliencia a oclusao", str(path), f"{delta:.2f} s", "retoma <= 2 s", status, "bloqueio de 3 s")


def summarize_fps(path: Path) -> ProtocolSummary:
    rows = read_rows(path)
    if not rows:
        return pending("Telemetria de FPS", path, "FPS medio", ">= 20 FPS")

    values = [parse_float(row.get("fps_current")) for row in rows if parse_float(row.get("fps_current")) > 0]
    if not values:
        return ProtocolSummary("Telemetria de FPS", str(path), "sem FPS valido", ">= 20 FPS", "atenção", "reexecutar por 60 s")

    mean = statistics.fmean(values)
    minimum = min(values)
    maximum = max(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    status = "ok" if mean >= 20.0 else "atenção"
    metric = f"media {mean:.2f}; dp {std:.2f}; min {minimum:.2f}; max {maximum:.2f}"
    return ProtocolSummary("Telemetria de FPS", str(path), metric, ">= 20 FPS", status, f"n={len(values)} frames")


def summarize_confusion(path: Path) -> ProtocolSummary:
    rows = read_rows(path)
    if not rows:
        return pending("Matriz de confusao", path, "Acuracia, precisao, recall, F1", ">= 50 amostras; acuracia >= 70%")

    row = rows[-1]
    samples = int(float(row.get("samples") or 0))
    accuracy = parse_float(row.get("accuracy")) * 100.0
    precision = parse_float(row.get("precision_macro")) * 100.0
    recall = parse_float(row.get("recall_macro")) * 100.0
    f1 = parse_float(row.get("f1_macro")) * 100.0
    status = "ok" if samples >= 50 and accuracy >= 70.0 else "atenção"
    metric = f"n={samples}; acc {accuracy:.1f}%; prec {precision:.1f}%; rec {recall:.1f}%; F1 {f1:.1f}%"
    return ProtocolSummary("Matriz de confusao", str(path), metric, ">= 50 amostras; acuracia >= 70%", status, "usar amostras reais na entrega")


def summarize_failover(path: Path) -> ProtocolSummary:
    rows = read_rows(path)
    if not rows:
        return pending("Tolerancia a falhas", path, "Retomada apos desconexao", "mensagem amigavel e retomada")

    read_values = [int(float(row.get("read_ok") or 0)) for row in rows]
    saw_failure = 0 in read_values
    recovered = False
    if saw_failure:
        first_failure = read_values.index(0)
        recovered = any(value == 1 for value in read_values[first_failure + 1 :])
    status = "ok" if saw_failure and recovered else "atenção"
    metric = "falha observada e recuperada" if recovered else "falha nao observada ou sem recuperacao"
    notes = "durante avaliacao, desconectar e reconectar fisicamente a webcam"
    return ProtocolSummary("Tolerancia a falhas", str(path), metric, "mensagem amigavel e retomada", status, notes)


def pending(protocol: str, path: Path, metric: str, target: str) -> ProtocolSummary:
    return ProtocolSummary(protocol, str(path), metric, target, "pendente", "arquivo ainda nao gerado")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def write_markdown(path: Path, summaries: list[ProtocolSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumo dos Protocolos",
        "",
        "| Protocolo | Evidencia | Metrica | Criterio | Status | Observacao |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in summaries:
        lines.append(
            f"| {item.protocol} | `{Path(item.evidence).name}` | {item.metric} | {item.target} | {item.status} | {item.notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, summaries: list[ProtocolSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["protocol", "evidence", "metric", "target", "status", "notes"])
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "protocol": item.protocol,
                    "evidence": item.evidence,
                    "metric": item.metric,
                    "target": item.target,
                    "status": item.status,
                    "notes": item.notes,
                }
            )


if __name__ == "__main__":
    main()
