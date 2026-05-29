from __future__ import annotations

import argparse
import csv
import html
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edutech_vision.config import AppConfig, ROOT_DIR, default_face_confidence
from edutech_vision.core.detection import FaceDetection, intersection_over_union, prioritize_detections
from edutech_vision.core.landmarks import FaceLandmarkDetector


DEFAULT_CONFIG = AppConfig()
BENCHMARK_DIR = ROOT_DIR / "assets" / "benchmarks"
WIDER_IMAGES = BENCHMARK_DIR / "widerface" / "WIDER_val" / "images"
WIDER_ANNOTATIONS = BENCHMARK_DIR / "widerface" / "wider_face_split" / "wider_face_val_bbx_gt.txt"
VIDEOS_DIR = BENCHMARK_DIR / "videos"
OUTPUT_ROOT = ROOT_DIR / "results" / "benchmark"
VARIANTS = ("base", "low_light", "bright", "blur", "distance", "occlusion", "no_face")


@dataclass(frozen=True)
class Annotation:
    bbox: tuple[float, float, float, float]
    blur: bool
    illumination: bool
    occlusion: bool
    pose: bool
    ignored: bool = False

    @property
    def size(self) -> str:
        side = min(self.bbox[2], self.bbox[3])
        if side < 32:
            return "small"
        if side < 96:
            return "medium"
        return "large"


@dataclass
class MetricBucket:
    truths: int = 0
    predictions: int = 0
    true_positives: int = 0
    landmark_true_positives: int = 0
    landmark_valid_predictions: int = 0
    ious: list[float] | None = None

    def __post_init__(self) -> None:
        if self.ious is None:
            self.ious = []

    @property
    def recall(self) -> float:
        return self.true_positives / self.truths if self.truths else 0.0

    @property
    def precision(self) -> float:
        return self.true_positives / self.predictions if self.predictions else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall else 0.0

    @property
    def mean_iou(self) -> float:
        return statistics.fmean(self.ious) if self.ious else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reproduzivel de deteccao/landmarks EduTech Vision.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--detector", choices=("mediapipe", "enhanced", "research"), required=True)
    parser.add_argument("--mode", choices=("individual", "plateia"), default="plateia")
    parser.add_argument("--max-images", type=int, default=0, help="Limite opcional para depuracao.")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Variantes separadas por virgula.")
    parser.add_argument("--confidence", type=float, help="Confianca minima. Padrao: melhor valor validado por modo.")
    parser.add_argument(
        "--max-faces",
        type=int,
        default=0,
        help="Maximo de faces no modo plateia. Padrao: melhor configuracao validada do app.",
    )
    args = parser.parse_args()

    ensure_corpus()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    invalid = sorted(set(variants) - set(VARIANTS))
    if invalid:
        raise SystemExit(f"Variantes invalidas: {', '.join(invalid)}")
    if args.confidence is None:
        args.confidence = default_face_confidence(args.mode)
    max_faces = 1 if args.mode == "individual" else args.max_faces or DEFAULT_CONFIG.max_audience_faces
    records = read_wider_annotations()
    records = select_records(records, args.suite, args.max_images, args.mode)
    output_dir = OUTPUT_ROOT / args.mode / args.detector / args.suite
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceLandmarkDetector(
        max_faces=max_faces,
        refine_landmarks=args.mode == "individual",
        detector_profile=args.detector,
        min_detection_confidence=args.confidence,
        running_mode="image",
    )
    try:
        results, buckets, timings, false_negatives, false_positives = evaluate_images(
            detector, records, variants, args.mode, max_faces
        )
        if args.mode == "individual":
            video_rows, video_timings = evaluate_individual_performance(detector, records, args.suite)
        else:
            video_rows, video_timings = evaluate_videos(detector, args.suite)
    finally:
        detector.close()

    write_rows(output_dir / "detections.csv", results)
    write_rows(output_dir / "video_metrics.csv", video_rows)
    write_metrics(output_dir, args, buckets, timings, video_timings, len(records), variants, max_faces)
    contact_sheet(false_negatives, output_dir / "false_negatives_contact_sheet.png", "Falsos negativos")
    contact_sheet(false_positives, output_dir / "false_positives_contact_sheet.png", "Falsos positivos")
    if args.detector == "enhanced":
        contact_sheet(false_negatives, OUTPUT_ROOT / "false_negatives_contact_sheet.png", "Falsos negativos")
        contact_sheet(false_positives, OUTPUT_ROOT / "false_positives_contact_sheet.png", "Falsos positivos")
    update_aggregate_metrics(output_dir / "metrics_by_detector.csv")
    summary = generate_summary()
    (OUTPUT_ROOT / "summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT_ROOT / "summary.html").write_text(
        "<html><body><pre>" + html.escape(summary) + "</pre></body></html>",
        encoding="utf-8",
    )
    overall = buckets["overall"]
    print(
        f"{args.mode}/{args.detector}/{args.suite}: "
        f"recall={overall.recall:.3f} precision={overall.precision:.3f} "
        f"f1={overall.f1:.3f} fps_960x540={1 / statistics.fmean(video_timings or timings):.1f}"
    )
    print(f"Relatorio: {OUTPUT_ROOT / 'summary.md'}")


def ensure_corpus() -> None:
    if not WIDER_IMAGES.exists() or not WIDER_ANNOTATIONS.exists():
        raise SystemExit("WIDER FACE ausente. Rode: python scripts/download_benchmarks.py")


def read_wider_annotations() -> list[tuple[Path, list[Annotation]]]:
    lines = WIDER_ANNOTATIONS.read_text(encoding="utf-8").splitlines()
    output: list[tuple[Path, list[Annotation]]] = []
    index = 0
    while index < len(lines):
        relative = lines[index].strip()
        index += 1
        count = int(lines[index].strip())
        index += 1
        boxes: list[Annotation] = []
        for _ in range(count):
            fields = [int(value) for value in lines[index].split()]
            index += 1
            if len(fields) < 10 or fields[2] <= 0 or fields[3] <= 0:
                continue
            boxes.append(
                Annotation(
                    bbox=tuple(float(value) for value in fields[:4]),
                    blur=fields[4] > 0,
                    illumination=fields[6] > 0,
                    occlusion=fields[8] > 0,
                    pose=fields[9] > 0,
                    ignored=fields[7] == 1,
                )
            )
        if boxes:
            output.append((WIDER_IMAGES / relative, boxes))
    return output


def select_records(
    records: list[tuple[Path, list[Annotation]]], suite: str, maximum: int, mode: str
) -> list[tuple[Path, list[Annotation]]]:
    if mode == "individual":
        records = [record for record in records if sum(not item.ignored for item in record[1]) == 1]
    if suite == "smoke":
        criteria = [
            lambda boxes: any(item.size == "small" for item in boxes),
            lambda boxes: len(boxes) >= 5,
            lambda boxes: any(item.occlusion for item in boxes),
            lambda boxes: any(item.illumination for item in boxes),
            lambda boxes: any(item.pose for item in boxes),
            lambda boxes: any(item.size == "large" for item in boxes),
        ]
        chosen: list[tuple[Path, list[Annotation]]] = []
        seen: set[Path] = set()
        for criterion in criteria:
            for record in records:
                if record[0] not in seen and criterion(record[1]):
                    chosen.append(record)
                    seen.add(record[0])
                    if sum(1 for item in chosen if criterion(item[1])) >= 10:
                        break
        records = chosen[:60]
    if maximum:
        records = records[:maximum]
    return records


def evaluate_images(
    detector: FaceLandmarkDetector,
    records: list[tuple[Path, list[Annotation]]],
    variants: tuple[str, ...],
    mode: str,
    max_faces: int,
) -> tuple[list[dict[str, object]], dict[str, MetricBucket], list[float], list[np.ndarray], list[np.ndarray]]:
    rows: list[dict[str, object]] = []
    buckets: dict[str, MetricBucket] = defaultdict(MetricBucket)
    timings: list[float] = []
    false_negatives: list[np.ndarray] = []
    false_positives: list[np.ndarray] = []
    for image_path, original_truth in records:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for variant in variants:
            frame, truths = apply_variant(image, original_truth, variant)
            selected_truths = select_truths(truths, frame.shape, mode, max_faces)
            selected_boxes = {item.bbox for item in selected_truths}
            ignored_truths = [item for item in truths if item.ignored or item.bbox not in selected_boxes]
            truths = selected_truths
            started = time.perf_counter()
            faces = detector.detect(frame)
            timings.append(time.perf_counter() - started)
            retained_faces = [
                face
                for face in faces
                if not any(intersection_over_union(ignored.bbox, face.bbox) >= 0.5 for ignored in ignored_truths)
            ]
            predictions = [tuple(float(value) for value in face.bbox) for face in retained_faces]
            valid_landmark_predictions = [
                tuple(float(value) for value in face.bbox) for face in retained_faces if face.points is not None
            ]
            landmark_valid_predictions = sum(face.points is not None for face in retained_faces)
            matches, unmatched_truth, unmatched_predictions = match_boxes(truths, predictions)
            landmark_matches, _, _ = match_boxes(truths, valid_landmark_predictions)
            categories = categorize(truths, variant)
            if variant == "no_face":
                update_bucket(buckets["stress:no_face"], 0, len(predictions), [], landmark_valid_predictions)
            else:
                update_bucket(
                    buckets["overall"],
                    len(truths),
                    len(predictions),
                    matches,
                    landmark_valid_predictions,
                    landmark_matches,
                )
            for category in categories:
                category_truths = [truths[index] for index in categories[category]]
                category_matches = [iou for index, iou in matches if index in categories[category]]
                update_bucket(buckets[category], len(category_truths), len(category_truths), category_matches)
            rows.append(
                {
                    "image": str(image_path.relative_to(WIDER_IMAGES)),
                    "variant": variant,
                    "truths": len(truths),
                    "predictions": len(predictions),
                    "landmark_valid_predictions": landmark_valid_predictions,
                    "landmark_true_positives": len(landmark_matches),
                    "true_positives": len(matches),
                    "false_negatives": len(unmatched_truth),
                    "false_positives": len(unmatched_predictions),
                    "mean_iou": f"{statistics.fmean(iou for _, iou in matches):.5f}" if matches else "0.00000",
                    "seconds": f"{timings[-1]:.6f}",
                }
            )
            if unmatched_truth and len(false_negatives) < 12:
                false_negatives.append(annotate_failure(frame, truths, predictions, "FN", unmatched_truth))
            if unmatched_predictions and len(false_positives) < 12:
                false_positives.append(annotate_failure(frame, truths, predictions, "FP", unmatched_predictions))
    return rows, buckets, timings, false_negatives, false_positives


def select_truths(truths: list[Annotation], frame_shape: tuple[int, ...], mode: str, max_faces: int) -> list[Annotation]:
    maximum = 1 if mode == "individual" else max_faces
    candidates = [FaceDetection(item.bbox, 1.0, "truth") for item in truths if not item.ignored]
    selected = prioritize_detections(candidates, frame_shape, maximum)
    selected_boxes = {item.bbox for item in selected}
    return [item for item in truths if not item.ignored and item.bbox in selected_boxes]


def apply_variant(image: np.ndarray, truths: list[Annotation], variant: str) -> tuple[np.ndarray, list[Annotation]]:
    if variant == "base":
        return image.copy(), truths
    if variant == "low_light":
        return cv2.convertScaleAbs(image, alpha=0.34, beta=0), truths
    if variant == "bright":
        return cv2.convertScaleAbs(image, alpha=1.25, beta=22), truths
    if variant == "blur":
        return cv2.GaussianBlur(image, (9, 9), 0), truths
    if variant == "occlusion":
        output = image.copy()
        for truth in truths:
            x, y, width, height = (int(value) for value in truth.bbox)
            cv2.rectangle(output, (x, y + height // 2), (x + width, y + height), (10, 10, 10), -1)
        return output, truths
    if variant == "no_face":
        output = image.copy()
        for truth in truths:
            x, y, width, height = (int(value) for value in truth.bbox)
            cv2.rectangle(output, (x, y), (x + width, y + height), (24, 24, 24), -1)
        return output, []
    if variant == "distance":
        height, width = image.shape[:2]
        scale = 0.55
        reduced = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        output = np.zeros_like(image)
        x_offset = (width - reduced.shape[1]) // 2
        y_offset = (height - reduced.shape[0]) // 2
        output[y_offset : y_offset + reduced.shape[0], x_offset : x_offset + reduced.shape[1]] = reduced
        transformed = [
            Annotation(
                bbox=(
                    truth.bbox[0] * scale + x_offset,
                    truth.bbox[1] * scale + y_offset,
                    truth.bbox[2] * scale,
                    truth.bbox[3] * scale,
                ),
                blur=truth.blur,
                illumination=truth.illumination,
                occlusion=truth.occlusion,
                pose=truth.pose,
                ignored=truth.ignored,
            )
            for truth in truths
        ]
        return output, transformed
    raise ValueError(variant)


def match_boxes(
    truths: list[Annotation], predictions: list[tuple[float, float, float, float]]
) -> tuple[list[tuple[int, float]], list[int], list[int]]:
    candidates = sorted(
        (
            (intersection_over_union(truth.bbox, predicted), truth_index, pred_index)
            for truth_index, truth in enumerate(truths)
            for pred_index, predicted in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_prediction: set[int] = set()
    matches: list[tuple[int, float]] = []
    for iou, truth_index, pred_index in candidates:
        if iou < 0.5:
            break
        if truth_index in matched_truth or pred_index in matched_prediction:
            continue
        matched_truth.add(truth_index)
        matched_prediction.add(pred_index)
        matches.append((truth_index, iou))
    return (
        matches,
        [index for index in range(len(truths)) if index not in matched_truth],
        [index for index in range(len(predictions)) if index not in matched_prediction],
    )


def categorize(truths: list[Annotation], variant: str) -> dict[str, list[int]]:
    categories: dict[str, list[int]] = defaultdict(list)
    for index, truth in enumerate(truths):
        categories[f"size:{truth.size}"].append(index)
        categories[f"variant:{variant}"].append(index)
        for name, active in [
            ("blur", truth.blur),
            ("low_light", truth.illumination),
            ("occlusion", truth.occlusion),
            ("pose", truth.pose),
        ]:
            if active:
                categories[f"attribute:{name}"].append(index)
    return categories


def update_bucket(
    bucket: MetricBucket,
    truths: int,
    predictions: int,
    matches: list[tuple[int, float]] | list[float],
    landmark_valid_predictions: int = 0,
    landmark_matches: list[tuple[int, float]] | None = None,
) -> None:
    ious = [item[1] if isinstance(item, tuple) else item for item in matches]
    bucket.truths += truths
    bucket.predictions += predictions
    bucket.true_positives += len(ious)
    bucket.landmark_valid_predictions += landmark_valid_predictions
    bucket.landmark_true_positives += len(landmark_matches or [])
    assert bucket.ious is not None
    bucket.ious.extend(ious)


def evaluate_videos(detector: FaceLandmarkDetector, suite: str) -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    timings: list[float] = []
    max_samples = 30 if suite == "smoke" else 180
    for path in sorted(VIDEOS_DIR.glob("*")):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_samples
        step = max(1, frame_count // max_samples)
        sampled = 0
        frame_index = 0
        while sampled < max_samples:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            performance_frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            started = time.perf_counter()
            faces = detector.detect(performance_frame)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            rows.append(
                {
                    "video": path.name,
                    "frame": frame_index,
                    "faces": len(faces),
                    "faces_with_landmarks": sum(face.points is not None for face in faces),
                    "seconds": f"{elapsed:.6f}",
                    "fps": f"{1 / elapsed:.3f}" if elapsed else "0.000",
                }
            )
            sampled += 1
            frame_index += step
        capture.release()
    return rows, timings


def evaluate_individual_performance(
    detector: FaceLandmarkDetector,
    records: list[tuple[Path, list[Annotation]]],
    suite: str,
) -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    timings: list[float] = []
    candidates = [
        path
        for path, truths in records
        if any(not truth.ignored and truth.size == "large" for truth in truths)
    ] or [path for path, _ in records]
    max_samples = 30 if suite == "smoke" else 180
    for index, path in enumerate((candidates * (max_samples // len(candidates) + 1))[:max_samples]):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        started = time.perf_counter()
        faces = detector.detect(frame)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        rows.append(
            {
                "video": f"single_face_960x540:{path.name}",
                "frame": index,
                "faces": len(faces),
                "faces_with_landmarks": sum(face.points is not None for face in faces),
                "seconds": f"{elapsed:.6f}",
                "fps": f"{1 / elapsed:.3f}" if elapsed else "0.000",
            }
        )
    return rows, timings


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(
    output_dir: Path,
    args: argparse.Namespace,
    buckets: dict[str, MetricBucket],
    image_timings: list[float],
    video_timings: list[float],
    images: int,
    variants: tuple[str, ...],
    max_faces: int,
) -> None:
    performance_timings = video_timings or image_timings
    fps = 1.0 / statistics.fmean(performance_timings) if performance_timings else 0.0
    minimum_fps = 1.0 / max(performance_timings) if performance_timings else 0.0
    image_fps = 1.0 / statistics.fmean(image_timings) if image_timings else 0.0
    rows: list[dict[str, object]] = []
    for category, bucket in sorted(buckets.items()):
        rows.append(
            {
                "mode": args.mode,
                "detector": args.detector,
                "suite": args.suite,
                "confidence": f"{args.confidence:.2f}",
                "max_faces": max_faces,
                "category": category,
                "truths": bucket.truths,
                "predictions": bucket.predictions,
                "true_positives": bucket.true_positives,
                "landmark_true_positives": bucket.landmark_true_positives,
                "landmark_valid_predictions": bucket.landmark_valid_predictions,
                "landmark_valid_rate": f"{bucket.landmark_valid_predictions / bucket.predictions:.5f}"
                if bucket.predictions
                else "0.00000",
                "landmark_recall": f"{bucket.landmark_true_positives / bucket.truths:.5f}" if bucket.truths else "0.00000",
                "recall": f"{bucket.recall:.5f}",
                "precision": f"{bucket.precision:.5f}" if category == "overall" else "",
                "f1": f"{bucket.f1:.5f}" if category == "overall" else "",
                "mean_iou": f"{bucket.mean_iou:.5f}",
                "fps_mean": f"{fps:.3f}",
                "fps_min": f"{minimum_fps:.3f}",
                "image_fps_native": f"{image_fps:.3f}",
                "images": images,
                "variants": ",".join(variants),
            }
        )
    write_rows(output_dir / "metrics_by_detector.csv", rows)
    write_rows(output_dir / "metrics_by_face_size.csv", [row for row in rows if str(row["category"]).startswith("size:")])


def update_aggregate_metrics(latest: Path) -> None:
    collected: list[dict[str, object]] = []
    for path in OUTPUT_ROOT.glob("*/*/*/metrics_by_detector.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            collected.extend(csv.DictReader(file))
    write_rows(OUTPUT_ROOT / "metrics_by_detector.csv", collected)
    sizes = [row for row in collected if str(row["category"]).startswith("size:")]
    write_rows(OUTPUT_ROOT / "metrics_by_face_size.csv", sizes)


def generate_summary() -> str:
    metrics_path = OUTPUT_ROOT / "metrics_by_detector.csv"
    if not metrics_path.exists():
        return "# Benchmark EduTech Vision\n\nNenhuma execucao registrada.\n"
    with metrics_path.open(newline="", encoding="utf-8") as file:
        all_rows = list(csv.DictReader(file))
    current_rows = [row for row in all_rows if row.get("confidence") and row.get("max_faces")]
    summary_source = current_rows or all_rows
    rows = [row for row in summary_source if row["category"] == "overall"]
    lines = [
        "# Benchmark EduTech Vision",
        "",
        "| Modo | Detector | Suite | Conf. | Max faces | Recall face | Recall landmarks | Precisao | F1 | Landmarks validos | FPS 960x540 | FPS min |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["mode"], item["suite"], item["detector"])):
        confidence = row.get("confidence") or "-"
        max_faces = row.get("max_faces") or "-"
        lines.append(
            f"| {row['mode']} | {row['detector']} | {row['suite']} | {confidence} | {max_faces} | {float(row['recall']):.3f} | "
            f"{float(row.get('landmark_recall') or row['recall']):.3f} | "
            f"{float(row['precision']):.3f} | {float(row['f1']):.3f} | {float(row.get('landmark_valid_rate') or 0.0):.3f} | "
            f"{float(row['fps_mean']):.1f} | "
            f"{float(row['fps_min']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gates Do Perfil Enhanced",
            "",
            "| Modo | Suite | Ganho recall geral | Ganho faces pequenas | Ganho distancia | Faces medias/grandes >= -0.02 | Precisao >= .85 | FPS minimo | Resultado |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    indexed = {(row["mode"], row["suite"], row["detector"], row["category"]): row for row in summary_source}
    for mode, suite in sorted({(row["mode"], row["suite"]) for row in rows}):
        baseline = indexed.get((mode, suite, "mediapipe", "overall"))
        enhanced = indexed.get((mode, suite, "enhanced", "overall"))
        if not baseline or not enhanced:
            continue
        small_baseline = indexed.get((mode, suite, "mediapipe", "size:small"))
        small_enhanced = indexed.get((mode, suite, "enhanced", "size:small"))
        distance_baseline = indexed.get((mode, suite, "mediapipe", "variant:distance"))
        distance_enhanced = indexed.get((mode, suite, "enhanced", "variant:distance"))
        medium_baseline = indexed.get((mode, suite, "mediapipe", "size:medium"))
        medium_enhanced = indexed.get((mode, suite, "enhanced", "size:medium"))
        large_baseline = indexed.get((mode, suite, "mediapipe", "size:large"))
        large_enhanced = indexed.get((mode, suite, "enhanced", "size:large"))
        recall_gain = float(enhanced["recall"]) - float(baseline["recall"])
        small_gain = metric_gain(small_enhanced, small_baseline, "recall")
        distance_gain = metric_gain(distance_enhanced, distance_baseline, "recall")
        medium_gain = metric_gain(medium_enhanced, medium_baseline, "recall")
        large_gain = metric_gain(large_enhanced, large_baseline, "recall")
        non_regression_ok = min(medium_gain, large_gain) >= -0.02
        precision_ok = float(enhanced["precision"]) >= 0.85
        fps_limit = 15.0 if mode == "individual" else 10.0
        fps_ok = float(enhanced["fps_mean"]) >= fps_limit
        recall_ok = max(small_gain, distance_gain) >= 0.20
        result = "PASS" if precision_ok and fps_ok and recall_ok and non_regression_ok else "REVER"
        lines.append(
            f"| {mode} | {suite} | {recall_gain:+.3f} | {small_gain:+.3f} | {distance_gain:+.3f} | "
            f"{'sim' if non_regression_ok else 'nao'} ({medium_gain:+.3f}/{large_gain:+.3f}) | "
            f"{'sim' if precision_ok else 'nao'} | {float(enhanced['fps_mean']):.1f} / {fps_limit:.0f} | {result} |"
        )
    lines.extend(
        [
            "",
            "## Gaps Observados",
            "",
            "- O baseline MediaPipe em frame inteiro perde faces pequenas/distantes; o ganho do perfil enhanced mede a correcao.",
            "- `Recall face` mede caixas localizadas; `recall landmarks` mede caixas que tambem permitem EAR/pose.",
            "- Faces detectadas sem landmarks confiaveis sao mostradas, mas nao entram em pose/engajamento.",
            "- O perfil research SCRFD e comparativo academico; a distribuicao padrao permanece YuNet.",
            "",
            "Criterios: ganho de pelo menos +0.20 em faces pequenas ou variante de distancia, regressao maxima de -0.02 em faces medias/grandes, precisao >= 0.85 e FPS em 960x540 suficiente para a demo.",
            "SCRFD/InsightFace e perfil de pesquisa non-commercial; YuNet e o perfil enhanced recomendado para distribuicao.",
        ]
    )
    return "\n".join(lines) + "\n"


def metric_gain(current: dict[str, str] | None, baseline: dict[str, str] | None, field: str) -> float:
    if not current or not baseline:
        return 0.0
    return float(current[field]) - float(baseline[field])


def annotate_failure(
    frame: np.ndarray,
    truths: list[Annotation],
    predictions: list[tuple[float, float, float, float]],
    label: str,
    indices: list[int],
) -> np.ndarray:
    output = frame.copy()
    boxes = [truths[index].bbox for index in indices] if label == "FN" else [predictions[index] for index in indices]
    for x, y, width, height in boxes:
        cv2.rectangle(output, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 2)
    cv2.putText(output, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return cv2.resize(output, (320, 220))


def contact_sheet(images: list[np.ndarray], path: Path, title: str) -> None:
    if not images:
        empty = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(empty, f"{title}: nenhum exemplo", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.imwrite(str(path), empty)
        return
    rows: list[np.ndarray] = []
    for start in range(0, len(images), 3):
        chunk = images[start : start + 3]
        while len(chunk) < 3:
            chunk.append(np.zeros_like(images[0]))
        rows.append(np.hstack(chunk))
    cv2.imwrite(str(path), np.vstack(rows))


if __name__ == "__main__":
    main()
