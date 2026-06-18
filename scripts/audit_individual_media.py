from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from edutech_vision.config import AppConfig
from edutech_vision.core.landmarks import FaceLandmarkDetector
from edutech_vision.modes.individual import BaselineCalibrator, IndividualMode, _angle_delta, _frontal_pitch_delta
from edutech_vision.utils.camera import resize_frame


ROOT_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT_DIR / "assets" / "benchmarks" / "individual"
OUTPUT_DIR = ROOT_DIR / "results" / "individual_media_audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


@dataclass(frozen=True)
class AuditRow:
    media_type: str
    file: str
    frame_index: int
    video_seconds: float
    face_detected: int
    tracking_quality: int
    quality_note: str
    ear: float | None
    left_ear: float | None
    right_ear: float | None
    mar: float | None
    yaw: float | None
    pitch: float | None
    yaw_delta: float | None
    pitch_delta: float | None
    baseline_samples: int
    baseline_calibrated: int
    calibrating: int
    front_facing: int
    neutral_expression: int
    passed: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita imagens e videos frontais no modo Individual.")
    parser.add_argument("--media-dir", type=Path, default=MEDIA_DIR, help="Pasta ou arquivo unico de midia.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--detector", choices=("mediapipe", "enhanced", "research"), default="enhanced")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--frame-step", type=int, default=1, help="1 processa todos os frames do video.")
    parser.add_argument("--max-video-seconds", type=float, default=0.0, help="0 processa videos inteiros.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = AppConfig(detector_profile=args.detector, face_confidence=args.confidence)
    rows: list[AuditRow] = []
    with FaceLandmarkDetector(
        max_faces=1,
        model_path=str(config.face_landmarker_model),
        detector_profile=args.detector,
        yunet_model_path=str(config.yunet_model),
        scrfd_model_path=str(config.scrfd_model),
        scrfd_input_size=config.scrfd_input_size,
        min_detection_confidence=args.confidence,
    ) as detector:
        for image_path in iter_media_paths(args.media_dir):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(audit_image(image_path, detector, config))
        for video_path in iter_media_paths(args.media_dir):
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                rows.extend(audit_video(video_path, detector, config, args.frame_step, args.max_video_seconds))

    csv_path = args.output_dir / "individual_media_audit.csv"
    write_rows(csv_path, rows)
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(build_summary(rows), encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"Resumo: {summary_path}")


def iter_media_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*"))


def audit_image(path: Path, detector: FaceLandmarkDetector, config: AppConfig) -> AuditRow:
    frame = cv2.imread(str(path))
    if frame is None:
        return empty_row("image", path, 0, 0.0, "imagem invalida")
    return analyze_frame("image", path, 0, 0.0, frame, detector, config, None)


def audit_video(
    path: Path,
    detector: FaceLandmarkDetector,
    config: AppConfig,
    frame_step: int,
    max_video_seconds: float,
) -> list[AuditRow]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [empty_row("video", path, 0, 0.0, "video invalido")]
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    calibrator = BaselineCalibrator(config.calibration_seconds)
    rows: list[AuditRow] = []
    frame_index = 0
    started = time.monotonic()
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        video_seconds = frame_index / fps if fps > 0 else 0.0
        if max_video_seconds > 0 and video_seconds > max_video_seconds:
            break
        if frame_index % max(1, frame_step) == 0:
            frame = resize_frame(frame, config.width, config.height)
            rows.append(analyze_frame("video", path, frame_index, video_seconds, frame, detector, config, calibrator))
        frame_index += 1
    capture.release()
    elapsed = max(1e-6, time.monotonic() - started)
    print(f"{path.name}: {len(rows)} amostras em {elapsed:.1f}s ({len(rows) / elapsed:.1f} fps processados)")
    return rows


def analyze_frame(
    media_type: str,
    path: Path,
    frame_index: int,
    video_seconds: float,
    frame,
    detector: FaceLandmarkDetector,
    config: AppConfig,
    calibrator: BaselineCalibrator | None,
) -> AuditRow:
    faces = detector.detect(frame)
    if not faces:
        return empty_row(media_type, path, frame_index, video_seconds, "face ausente")
    face = faces[0]
    if face.points is None:
        return empty_row(media_type, path, frame_index, video_seconds, "landmarks insuficientes", face_detected=1)

    analysis = IndividualMode._compute_analysis(face.points, frame)
    baseline_samples = 0
    baseline_calibrated = 0
    calibrating = 0
    if calibrator is not None and analysis.quality_ok:
        calibrator.update(analysis.metrics, calibrator.started_at + video_seconds)
        baseline = calibrator.baseline
        baseline_samples = baseline.samples
        baseline_calibrated = int(baseline.calibrated)
        calibrating = int(calibrator.active(calibrator.started_at + video_seconds))
        if baseline.calibrated:
            yaw_delta = _angle_delta(analysis.metrics.yaw, baseline.yaw)
            pitch_delta = _angle_delta(analysis.metrics.pitch, baseline.pitch)
        else:
            yaw_delta = _angle_delta(analysis.metrics.yaw, 0.0)
            pitch_delta = _frontal_pitch_delta(analysis.metrics.pitch)
    else:
        baseline = calibrator.baseline if calibrator is not None else config_baseline()
        baseline_samples = baseline.samples
        baseline_calibrated = int(baseline.calibrated)
        calibrating = int(calibrator.active(calibrator.started_at + video_seconds)) if calibrator is not None else 0
        if baseline.calibrated:
            yaw_delta = _angle_delta(analysis.metrics.yaw, baseline.yaw)
            pitch_delta = _angle_delta(analysis.metrics.pitch, baseline.pitch)
        else:
            yaw_delta = _angle_delta(analysis.metrics.yaw, 0.0)
            pitch_delta = _frontal_pitch_delta(analysis.metrics.pitch)

    front_facing = int(analysis.quality_ok and abs(yaw_delta) <= config.attention_yaw_tolerance and abs(pitch_delta) <= config.posture_pitch_tolerance)
    neutral_expression = int(analysis.quality_ok and analysis.metrics.ear >= 0.14 and analysis.metrics.mar <= 0.28)
    passed = int(front_facing and neutral_expression)
    return AuditRow(
        media_type=media_type,
        file=relative_project_path(path),
        frame_index=frame_index,
        video_seconds=video_seconds,
        face_detected=1,
        tracking_quality=int(analysis.quality_ok),
        quality_note=analysis.quality_note,
        ear=analysis.metrics.ear,
        left_ear=analysis.left_ear,
        right_ear=analysis.right_ear,
        mar=analysis.metrics.mar,
        yaw=analysis.metrics.yaw,
        pitch=analysis.metrics.pitch,
        yaw_delta=yaw_delta,
        pitch_delta=pitch_delta,
        baseline_samples=baseline_samples,
        baseline_calibrated=baseline_calibrated,
        calibrating=calibrating,
        front_facing=front_facing,
        neutral_expression=neutral_expression,
        passed=passed,
    )


def config_baseline():
    return BaselineCalibrator(0.0).baseline


def empty_row(
    media_type: str,
    path: Path,
    frame_index: int,
    video_seconds: float,
    note: str,
    *,
    face_detected: int = 0,
) -> AuditRow:
    return AuditRow(
        media_type=media_type,
        file=relative_project_path(path),
        frame_index=frame_index,
        video_seconds=video_seconds,
        face_detected=face_detected,
        tracking_quality=0,
        quality_note=note,
        ear=None,
        left_ear=None,
        right_ear=None,
        mar=None,
        yaw=None,
        pitch=None,
        yaw_delta=None,
        pitch_delta=None,
        baseline_samples=0,
        baseline_calibrated=0,
        calibrating=0,
        front_facing=0,
        neutral_expression=0,
        passed=0,
    )


def relative_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def write_rows(path: Path, rows: list[AuditRow]) -> None:
    fields = list(AuditRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(getattr(row, field)) for field in fields})


def format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.5f}"
    if value is None:
        return ""
    return value


def build_summary(rows: list[AuditRow]) -> str:
    lines = ["# Auditoria Individual - Midia Frontal", ""]
    if not rows:
        return "\n".join(lines + ["Nenhuma midia auditada.", ""])

    for media_type in ("image", "video"):
        subset = [row for row in rows if row.media_type == media_type]
        if not subset:
            continue
        lines.extend(
            [
                f"## {media_type.title()}",
                "",
                f"- Amostras: {len(subset)}",
                f"- Face detectada: {rate(subset, 'face_detected'):.1f}%",
                f"- Metricas confiaveis: {rate(subset, 'tracking_quality'):.1f}%",
                f"- Frontal: {rate(subset, 'front_facing'):.1f}%",
                f"- Expressao neutra: {rate(subset, 'neutral_expression'):.1f}%",
                f"- Passou: {rate(subset, 'passed'):.1f}%",
                f"- EAR mediano: {median(row.ear for row in subset):.3f}",
                f"- MAR mediano: {median(row.mar for row in subset):.3f}",
                f"- |Yaw delta| mediano: {median_abs(row.yaw_delta for row in subset):.1f} deg",
                f"- |Pitch delta| mediano: {median_abs(row.pitch_delta for row in subset):.1f} deg",
                "",
            ]
        )

    lines.extend(["## Por Arquivo", "", "| Arquivo | Tipo | Amostras | Face % | Qualidade % | Frontal % | Passou % |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for file_name in sorted({row.file for row in rows}):
        subset = [row for row in rows if row.file == file_name]
        lines.append(
            f"| `{file_name}` | {subset[0].media_type} | {len(subset)} | "
            f"{rate(subset, 'face_detected'):.1f} | {rate(subset, 'tracking_quality'):.1f} | "
            f"{rate(subset, 'front_facing'):.1f} | {rate(subset, 'passed'):.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def rate(rows: list[AuditRow], field: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(int(getattr(row, field)) for row in rows) / len(rows)


def median(values) -> float:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else 0.0


def median_abs(values) -> float:
    clean = [abs(float(value)) for value in values if value is not None]
    return statistics.median(clean) if clean else 0.0


if __name__ == "__main__":
    main()
