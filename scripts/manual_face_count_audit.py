from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edutech_vision.config import ROOT_DIR
from edutech_vision.core.aggregation import EngagementWindow
from edutech_vision.core.landmarks import FaceLandmarkDetector
from edutech_vision.core.models import FaceLandmarks
from edutech_vision.core.pose import PoseResult, estimate_head_pose


COUNT_FILE = ROOT_DIR / "docs" / "manual_face_audit" / "manual_counts.csv"
MANIFEST_FILE = ROOT_DIR / "assets" / "benchmarks" / "manifest.json"
VIDEOS_DIR = ROOT_DIR / "assets" / "benchmarks" / "videos"
OUTPUT_DIR = ROOT_DIR / "docs" / "manual_face_audit"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
LOG_DIR = OUTPUT_DIR / "realtime_logs"
WINDOW_RADIUS_SECONDS = 1.0


@dataclass(frozen=True)
class ManualSample:
    id: str
    video: str
    timestamp_s: float
    manual_faces: int
    manual_notes: str


@dataclass(frozen=True)
class AuditConfig:
    detector: str
    run_label: str
    max_faces: int
    confidence: float
    scrfd_input_size: int
    window_seconds: float
    calibration_seconds: float
    yaw_tolerance: float
    pitch_tolerance: float
    report_width: int


@dataclass
class VideoStageCalibrator:
    duration_seconds: float
    default_yaw: float = 0.0
    default_pitch: float = 0.0
    yaw_samples: list[float] | None = None
    pitch_samples: list[float] | None = None

    def __post_init__(self) -> None:
        self.yaw_samples = []
        self.pitch_samples = []

    def active(self, timestamp_s: float) -> bool:
        return timestamp_s < self.duration_seconds

    def update(self, pose: PoseResult, timestamp_s: float) -> None:
        if not self.active(timestamp_s):
            return
        assert self.yaw_samples is not None
        assert self.pitch_samples is not None
        self.yaw_samples.append(pose.yaw)
        self.pitch_samples.append(pose.pitch)

    @property
    def yaw(self) -> float:
        assert self.yaw_samples is not None
        if len(self.yaw_samples) < 5:
            return self.default_yaw
        return float(np.median(np.array(self.yaw_samples, dtype=np.float32)))

    @property
    def pitch(self) -> float:
        assert self.pitch_samples is not None
        if len(self.pitch_samples) < 5:
            return self.default_pitch
        return float(np.median(np.array(self.pitch_samples, dtype=np.float32)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auditoria manual-vs-app para contagem de rostos em videos completos."
    )
    parser.add_argument("--detector", choices=("mediapipe", "enhanced", "research"), default="enhanced")
    parser.add_argument("--run-label", help="Sufixo dos arquivos gerados. Padrao deriva do detector/config.")
    parser.add_argument("--max-faces", type=int, default=24)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--scrfd-input-size", type=int, default=640)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--yaw-tolerance", type=float, default=30.0)
    parser.add_argument("--pitch-tolerance", type=float, default=20.0)
    parser.add_argument("--report-width", type=int, default=900)
    parser.add_argument("--count-file", type=Path, default=COUNT_FILE)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest-file", type=Path, default=MANIFEST_FILE)
    args = parser.parse_args()

    configure_paths(args.count_file, args.videos_dir, args.output_dir, args.manifest_file)
    config = AuditConfig(
        detector=args.detector,
        run_label=args.run_label or default_run_label(args.detector, args.confidence, args.scrfd_input_size, args.max_faces),
        max_faces=args.max_faces,
        confidence=args.confidence,
        scrfd_input_size=args.scrfd_input_size,
        window_seconds=args.window_seconds,
        calibration_seconds=args.calibration_seconds,
        yaw_tolerance=args.yaw_tolerance,
        pitch_tolerance=args.pitch_tolerance,
        report_width=args.report_width,
    )
    samples = read_samples()
    prepare_output()
    grouped_samples = group_samples(samples)

    log_paths: dict[str, Path] = {}
    for video_name, video_samples in grouped_samples.items():
        log_paths[video_name] = run_full_video(video_name, video_samples, config)

    rows = analyze_logs(samples, log_paths, config)
    write_csv(OUTPUT_DIR / f"detector_comparison_{config.run_label}.csv", rows)
    write_report(rows, log_paths, config)
    write_profile_summary()
    print(f"Auditoria gerada: {OUTPUT_DIR / 'report.md'}")


def configure_paths(count_file: Path, videos_dir: Path, output_dir: Path, manifest_file: Path) -> None:
    global COUNT_FILE, MANIFEST_FILE, VIDEOS_DIR, OUTPUT_DIR, COMPARISON_DIR, LOG_DIR
    COUNT_FILE = count_file if count_file.is_absolute() else ROOT_DIR / count_file
    VIDEOS_DIR = videos_dir if videos_dir.is_absolute() else ROOT_DIR / videos_dir
    OUTPUT_DIR = output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir
    MANIFEST_FILE = manifest_file if manifest_file.is_absolute() else ROOT_DIR / manifest_file
    COMPARISON_DIR = OUTPUT_DIR / "comparison"
    LOG_DIR = OUTPUT_DIR / "realtime_logs"


def read_samples() -> list[ManualSample]:
    with COUNT_FILE.open("r", encoding="utf-8", newline="") as file:
        return [
            ManualSample(
                id=row["id"],
                video=row["video"],
                timestamp_s=float(row["timestamp_s"]),
                manual_faces=int(row["manual_faces"]),
                manual_notes=row["manual_notes"],
            )
            for row in csv.DictReader(file)
        ]


def default_run_label(detector: str, confidence: float, scrfd_input_size: int, max_faces: int) -> str:
    if detector == "research" and scrfd_input_size != 640:
        return f"{detector}{scrfd_input_size}_c{int(confidence * 100):02d}_m{max_faces}"
    if detector != "mediapipe":
        return f"{detector}_c{int(confidence * 100):02d}_m{max_faces}"
    return f"{detector}_m{max_faces}"


def prepare_output() -> None:
    for directory in (COMPARISON_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def group_samples(samples: list[ManualSample]) -> dict[str, list[ManualSample]]:
    grouped: dict[str, list[ManualSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.video, []).append(sample)
    return grouped


def run_full_video(video_name: str, samples: list[ManualSample], config: AuditConfig) -> Path:
    video_path = VIDEOS_DIR / video_name
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video nao abriu: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sample_frames = map_samples_to_frames(samples, source_fps)
    log_path = LOG_DIR / f"{Path(video_name).stem}_{config.run_label}.csv"
    fields = [
        "video",
        "frame_index",
        "video_seconds",
        "wall_seconds",
        "processing_seconds",
        "throughput_fps",
        "source_fps",
        "faces",
        "faces_with_pose",
        "attentive",
        "engagement_percent",
        "faces_window_mean",
        "pose_window_mean",
        "attentive_window_mean",
        "detector",
        "run_label",
        "notes",
    ]
    detector = FaceLandmarkDetector(
        max_faces=config.max_faces,
        refine_landmarks=False,
        detector_profile=config.detector,
        min_detection_confidence=config.confidence,
        scrfd_input_size=config.scrfd_input_size,
    )
    detection_window = EngagementWindow(config.window_seconds)
    pose_window = EngagementWindow(config.window_seconds)
    calibrator = VideoStageCalibrator(config.calibration_seconds)
    started_wall = time.perf_counter()
    frame_index = 0

    print(f"Rodando video inteiro: {video_name} ({total_frames} frames, {source_fps:.2f} FPS)")
    try:
        with log_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                video_seconds = frame_index / source_fps
                detection_started = time.perf_counter()
                faces = detector.detect(frame)
                processing_seconds = time.perf_counter() - detection_started
                posed_count, attentive_count = summarize_audience_frame(frame, faces, calibrator, video_seconds, config)
                detection_window.update(video_seconds, len(faces), attentive_count)
                window_engagement = pose_window.update(video_seconds, posed_count, attentive_count)
                throughput = 1.0 / processing_seconds if processing_seconds > 0 else 0.0

                if frame_index in sample_frames:
                    for sample in sample_frames[frame_index]:
                        write_sample_images(sample, frame, faces, video_seconds, config)

                writer.writerow(
                    {
                        "video": video_name,
                        "frame_index": frame_index,
                        "video_seconds": f"{video_seconds:.3f}",
                        "wall_seconds": f"{time.perf_counter() - started_wall:.3f}",
                        "processing_seconds": f"{processing_seconds:.6f}",
                        "throughput_fps": f"{throughput:.3f}",
                        "source_fps": f"{source_fps:.3f}",
                        "faces": len(faces),
                        "faces_with_pose": posed_count,
                        "attentive": attentive_count,
                        "engagement_percent": f"{window_engagement:.3f}",
                        "faces_window_mean": f"{detection_window.mean_faces:.3f}",
                        "pose_window_mean": f"{pose_window.mean_faces:.3f}",
                        "attentive_window_mean": f"{pose_window.mean_attentive:.3f}",
                        "detector": config.detector,
                        "run_label": config.run_label,
                        "notes": "full_video_realtime_sequence",
                    }
                )
                frame_index += 1
    finally:
        detector.close()
        capture.release()
    print(f"Log temporal: {log_path.relative_to(ROOT_DIR)}")
    return log_path


def map_samples_to_frames(samples: list[ManualSample], source_fps: float) -> dict[int, list[ManualSample]]:
    output: dict[int, list[ManualSample]] = {}
    for sample in samples:
        frame_index = max(0, int(round(sample.timestamp_s * source_fps)))
        output.setdefault(frame_index, []).append(sample)
    return output


def summarize_audience_frame(
    frame: np.ndarray,
    faces: list[FaceLandmarks],
    calibrator: VideoStageCalibrator,
    video_seconds: float,
    config: AuditConfig,
) -> tuple[int, int]:
    posed_count = 0
    attentive_count = 0
    for face in faces:
        if face.points is None:
            continue
        pose = estimate_head_pose(face.points, frame.shape)
        if not pose.success:
            continue
        calibrator.update(pose, video_seconds)
        posed_count += 1
        yaw_ok = abs(pose.yaw - calibrator.yaw) <= config.yaw_tolerance
        pitch_ok = abs(pose.pitch - calibrator.pitch) <= config.pitch_tolerance
        attentive_count += int(yaw_ok and pitch_ok)
    return posed_count, attentive_count


def write_sample_images(
    sample: ManualSample,
    frame: np.ndarray,
    faces: list[FaceLandmarks],
    actual_seconds: float,
    config: AuditConfig,
) -> None:
    comparison_path = COMPARISON_DIR / f"{sample.id}_{config.run_label}_comparison.png"
    annotated = draw_predictions(frame, faces, sample, actual_seconds)
    cv2.imwrite(str(comparison_path), comparison_image(frame, annotated, sample, len(faces), actual_seconds, config))


def draw_predictions(
    frame: np.ndarray,
    faces: list[FaceLandmarks],
    sample: ManualSample,
    actual_seconds: float,
) -> np.ndarray:
    output = frame.copy()
    for index, face in enumerate(faces, start=1):
        x, y, width, height = face.bbox
        color = (80, 220, 80) if face.points is not None else (0, 210, 255)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 3)
        cv2.putText(output, f"D{index}", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    title = (
        f"{sample.id} | alvo={sample.timestamp_s:.2f}s real={actual_seconds:.2f}s | "
        f"manual={sample.manual_faces} detector={len(faces)}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (20, 20, 20), -1)
    cv2.putText(output, title[:150], (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)
    return output


def resize_for_report(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def comparison_image(
    raw: np.ndarray,
    annotated: np.ndarray,
    sample: ManualSample,
    exact_count: int,
    actual_seconds: float,
    config: AuditConfig,
) -> np.ndarray:
    left = resize_for_report(raw, config.report_width)
    right = resize_for_report(annotated, config.report_width)
    target_height = max(left.shape[0], right.shape[0])
    left = pad_to_height(left, target_height)
    right = pad_to_height(right, target_height)
    combined = np.hstack([left, right])
    header = np.full((100, combined.shape[1], 3), 245, dtype=np.uint8)
    text = (
        f"{sample.id} | {sample.video} | alvo={sample.timestamp_s:.2f}s real={actual_seconds:.2f}s | "
        f"manual={sample.manual_faces} | detector={exact_count}"
    )
    cv2.putText(header, text[:155], (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2)
    cv2.putText(header, sample.manual_notes[:155], (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
    return np.vstack([header, combined])


def pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), dtype=frame.dtype)
    return np.vstack([frame, pad])


def analyze_logs(samples: list[ManualSample], log_paths: dict[str, Path], config: AuditConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    logs = {video: read_log(path) for video, path in log_paths.items()}
    for sample in samples:
        video_rows = logs[sample.video]
        nearest = min(video_rows, key=lambda row: abs(float(row["video_seconds"]) - sample.timestamp_s))
        window_rows = [
            row
            for row in video_rows
            if abs(float(row["video_seconds"]) - sample.timestamp_s) <= WINDOW_RADIUS_SECONDS
        ]
        window_counts = [int(row["faces"]) for row in window_rows]
        window_pose_counts = [int(row["faces_with_pose"]) for row in window_rows]
        exact_count = int(nearest["faces"])
        exact_delta = exact_count - sample.manual_faces
        window_hits = sum(1 for value in window_counts if value == sample.manual_faces)
        window_hit_ratio = window_hits / len(window_counts) if window_counts else 0.0
        median_count = statistics.median(window_counts) if window_counts else exact_count
        rows.append(
            {
                "id": sample.id,
                "video": sample.video,
                "timestamp_s": f"{sample.timestamp_s:.2f}",
                "matched_video_seconds": f"{float(nearest['video_seconds']):.3f}",
                "manual_faces": sample.manual_faces,
                "detected_exact": exact_count,
                "pose_exact": int(nearest["faces_with_pose"]),
                "delta_exact": exact_delta,
                "window_min": min(window_counts) if window_counts else exact_count,
                "window_median": f"{median_count:.1f}",
                "window_max": max(window_counts) if window_counts else exact_count,
                "window_hit_ratio": f"{window_hit_ratio:.3f}",
                "window_pose_min": min(window_pose_counts) if window_pose_counts else int(nearest["faces_with_pose"]),
                "window_pose_max": max(window_pose_counts) if window_pose_counts else int(nearest["faces_with_pose"]),
                "processing_fps": nearest["throughput_fps"],
                "comparison_image": (COMPARISON_DIR / f"{sample.id}_{config.run_label}_comparison.png")
                .relative_to(OUTPUT_DIR)
                .as_posix(),
                "manual_notes": sample.manual_notes,
            }
        )
    return rows


def read_log(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], log_paths: dict[str, Path], config: AuditConfig) -> None:
    exact_errors = [abs(int(row["delta_exact"])) for row in rows]
    exact_matches = sum(1 for row in rows if int(row["delta_exact"]) == 0)
    within_one = sum(1 for row in rows if abs(int(row["delta_exact"])) <= 1)
    mean_abs_error = statistics.fmean(exact_errors) if exact_errors else 0.0
    manifest = read_manifest()
    lines = [
        "# Auditoria Manual De Contagem De Rostos Em Videos",
        "",
        "## Como O Teste Foi Rodado",
        "",
        "Cada arquivo de video foi processado do primeiro ao ultimo frame, uma unica vez, preservando a ordem temporal. "
        "O log em `realtime_logs/` registra a resposta do pipeline em cada frame: tempo do video, tempo de processamento, "
        "faces detectadas, faces com pose valida, atentos e medias da janela de plateia.",
        "",
        f"Detector testado: `{config.detector}` (`run_label={config.run_label}`) com `max_faces={config.max_faces}`, "
        f"confianca `{config.confidence:.2f}`, SCRFD `{config.scrfd_input_size}px` "
        f"e janela de plateia `{config.window_seconds:.1f}s`.",
        "",
        "Resumo comparativo entre perfis: [`profile_summary.md`](profile_summary.md).",
        "",
        "## Criterio Manual",
        "",
        "Contamos apenas rostos avaliaveis: face ou perfil humano visivel o bastante para uma caixa facial. "
        "Nao entram nucas, pessoas de costas, silhuetas estouradas pela janela, cabecas cobertas sem face aparente "
        "ou rostos muito cortados pela borda do frame.",
        "",
        "## Videos",
        "",
    ]
    for video_name, path in log_paths.items():
        source = manifest.get(video_name, {})
        rel_log = path.relative_to(OUTPUT_DIR).as_posix()
        lines.append(
            f"- `{video_name}`: log completo [`{rel_log}`]({rel_log}); "
            f"licenca {source.get('license', 'n/d')}; fonte {source.get('source_url', 'n/d')}."
        )
    lines.extend(
        [
            "",
            "## Resultado Por Frame Manual",
            "",
            "| ID | Manual | Detector | Delta | Pose valida | Janela +/-1s | Hit ratio +/-1s | FPS | Imagem |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['id']}` | {row['manual_faces']} | {row['detected_exact']} | {row['delta_exact']} | "
            f"{row['pose_exact']} | {row['window_min']} / {row['window_median']} / {row['window_max']} | "
            f"{row['window_hit_ratio']} | "
            f"{row['processing_fps']} | [comparar]({row['comparison_image']}) |"
        )
    lines.extend(
        [
            "",
            "## Sintese",
            "",
            f"- Frames auditados manualmente: {len(rows)}.",
            f"- Acertos exatos de contagem: {exact_matches}/{len(rows)}.",
            f"- Casos com erro maximo de 1 rosto: {within_one}/{len(rows)}.",
            f"- Erro absoluto medio no frame exato: {mean_abs_error:.2f} rosto(s).",
            "",
            "## Leitura Dos Gaps",
            "",
        ]
    )
    lines.extend(gap_notes(rows))
    lines.extend(
        [
            "",
            "## Observacoes Manuais",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['id']}`: {row['manual_notes']}")
    lines.extend(["", "## Prints Comparativos", ""])
    for row in rows:
        lines.append(f"### {row['id']}")
        lines.append("")
        lines.append(f"![{row['id']}]({row['comparison_image']})")
        lines.append("")
    report_text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / f"report_{config.run_label}.md").write_text(report_text, encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(report_text, encoding="utf-8")


def read_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST_FILE.exists():
        return {}
    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {
        str(entry["name"]): entry
        for entry in data.get("entries", [])
        if str(entry.get("name", "")).lower().endswith((".webm", ".mp4", ".mov", ".mkv"))
    }


def write_profile_summary() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(OUTPUT_DIR.glob("detector_comparison_*.csv")):
        label = path.stem.removeprefix("detector_comparison_")
        with path.open("r", encoding="utf-8", newline="") as file:
            samples = list(csv.DictReader(file))
        if not samples:
            continue
        deltas = [int(row["delta_exact"]) for row in samples]
        manual_in_window = [
            int(row["window_min"]) <= int(row["manual_faces"]) <= int(row["window_max"])
            for row in samples
        ]
        fps_values: list[float] = []
        for log_path in LOG_DIR.glob(f"*_{label}.csv"):
            with log_path.open("r", encoding="utf-8", newline="") as file:
                fps_values.extend(float(row["throughput_fps"]) for row in csv.DictReader(file))
        rows.append(
            {
                "run_label": label,
                "samples": len(samples),
                "exact_hits": sum(delta == 0 for delta in deltas),
                "within_one": sum(abs(delta) <= 1 for delta in deltas),
                "mean_abs_error": f"{statistics.fmean(abs(delta) for delta in deltas):.3f}",
                "manual_inside_window": sum(manual_in_window),
                "fps_mean": f"{statistics.fmean(fps_values):.1f}" if fps_values else "0.0",
                "fps_p10": f"{statistics.quantiles(fps_values, n=10)[0]:.1f}" if len(fps_values) >= 10 else "0.0",
            }
        )
    if not rows:
        return
    rows.sort(key=lambda row: (float(row["mean_abs_error"]), -int(row["exact_hits"]), -float(row["fps_mean"])))
    write_csv(OUTPUT_DIR / "profile_summary.csv", rows)
    lines = [
        "# Resumo Comparativo Dos Perfis",
        "",
        "Todos os perfis abaixo foram avaliados em videos completos, com comparacao contra os mesmos frames contados manualmente.",
        "",
        "| Perfil | Acertos | <=1 rosto | Erro medio | Manual dentro da janela +/-1s | FPS medio | FPS p10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_label']}` | {row['exact_hits']}/{row['samples']} | {row['within_one']}/{row['samples']} | "
            f"{row['mean_abs_error']} | {row['manual_inside_window']}/{row['samples']} | "
            f"{row['fps_mean']} | {row['fps_p10']} |"
        )
    best = rows[0]
    lines.extend(
        [
            "",
            "## Leitura",
            "",
            f"- Melhor perfil nesta auditoria: `{best['run_label']}`.",
            "- O criterio principal e menor erro absoluto medio; FPS entra como criterio secundario.",
            "- A janela +/-1s mostra se a resposta correta apareceu durante a sequencia temporal ao redor do frame manual.",
            "",
        ]
    )
    (OUTPUT_DIR / "profile_summary.md").write_text("\n".join(lines), encoding="utf-8")


def gap_notes(rows: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    misses = [row for row in rows if int(row["delta_exact"]) < 0]
    overcounts = [row for row in rows if int(row["delta_exact"]) > 0]
    unstable = [row for row in rows if float(row["window_hit_ratio"]) < 0.5]
    if misses:
        ids = ", ".join(f"`{row['id']}`" for row in misses)
        notes.append(f"- Subcontagem observada em {ids}; revisar rostos pequenos, perfil forte e oclusao.")
    if overcounts:
        ids = ", ".join(f"`{row['id']}`" for row in overcounts)
        notes.append(f"- Supercontagem observada em {ids}; revisar pessoas cortadas, falsos positivos e caixas duplicadas.")
    if unstable:
        ids = ", ".join(f"`{row['id']}`" for row in unstable)
        notes.append(f"- Contagem instavel na janela temporal em {ids}; isso afeta a experiencia ao vivo mesmo quando o frame isolado acerta.")
    if not notes:
        notes.append("- Nenhum gap grave apareceu nos frames auditados; manter estes videos como regressao real-time.")
    return notes


if __name__ == "__main__":
    main()
