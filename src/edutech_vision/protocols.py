from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from edutech_vision.config import AppConfig, parse_source
from edutech_vision.core.landmarks import FaceLandmarkDetector
from edutech_vision.utils.camera import ResilientVideoSource, blank_frame
from edutech_vision.utils.fps import SlidingFPS
from edutech_vision.utils.logging import CsvLogger, timestamp_iso


def run_fps_protocol(
    mode: str,
    duration: float,
    camera: str,
    video: str | None,
    output: Path,
) -> None:
    config = AppConfig(source=parse_source(camera, video))
    max_faces = 1 if mode == "individual" else config.max_audience_faces
    camera_source = ResilientVideoSource(config.source, config.width, config.height)
    detector = create_detector(config, mode, max_faces)
    fps = SlidingFPS(config.fps_window_seconds)
    fields = ["timestamp", "seconds", "mode", "faces_detected", "fps_current", "fps_mean", "fps_std", "fps_min", "fps_max"]
    start = time.monotonic()

    try:
        with CsvLogger(output, fields) as logger:
            while time.monotonic() - start < duration:
                now = time.monotonic()
                ok, frame = camera_source.read()
                stats = fps.update(now)
                faces_count = 0
                if ok and frame is not None:
                    faces_count = len(detector.detect(frame))
                    _draw_protocol_text(frame, f"FPS protocolo | faces={faces_count} | q sai")
                    cv2.imshow("Protocolo FPS", frame)
                else:
                    cv2.imshow("Protocolo FPS", blank_frame(config.width, config.height, "Fonte indisponivel. Tentando reconectar..."))

                logger.write(
                    timestamp=timestamp_iso(),
                    seconds=f"{now - start:.3f}",
                    mode=mode,
                    faces_detected=faces_count,
                    fps_current=f"{stats.current:.3f}",
                    fps_mean=f"{stats.mean:.3f}",
                    fps_std=f"{stats.std:.3f}",
                    fps_min=f"{stats.minimum:.3f}",
                    fps_max=f"{stats.maximum:.3f}",
                )
                if _quit_pressed():
                    break
    finally:
        detector.close()
        camera_source.release()
        cv2.destroyAllWindows()


def run_lighting_protocol(
    mode: str,
    stage_duration: float,
    camera: str,
    video: str | None,
    output: Path,
) -> None:
    config = AppConfig(source=parse_source(camera, video))
    max_faces = 1 if mode == "individual" else config.max_audience_faces
    camera_source = ResilientVideoSource(config.source, config.width, config.height)
    detector = create_detector(config, mode, max_faces)
    fps = SlidingFPS(config.fps_window_seconds)
    stages = ("baixa", "media", "alta")
    fields = ["timestamp", "seconds", "stage", "mode", "faces_detected", "fps_current", "fps_mean"]
    start = time.monotonic()

    try:
        with CsvLogger(output, fields) as logger:
            for stage in stages:
                stage_start = time.monotonic()
                while time.monotonic() - stage_start < stage_duration:
                    now = time.monotonic()
                    ok, frame = camera_source.read()
                    stats = fps.update(now)
                    faces_count = 0
                    if ok and frame is not None:
                        faces_count = len(detector.detect(frame))
                        _draw_protocol_text(frame, f"Iluminacao: {stage} | ajuste ambiente | q sai")
                        cv2.imshow("Protocolo Iluminacao", frame)
                    else:
                        cv2.imshow("Protocolo Iluminacao", blank_frame(config.width, config.height, "Fonte indisponivel. Tentando reconectar..."))

                    logger.write(
                        timestamp=timestamp_iso(),
                        seconds=f"{now - start:.3f}",
                        stage=stage,
                        mode=mode,
                        faces_detected=faces_count,
                        fps_current=f"{stats.current:.3f}",
                        fps_mean=f"{stats.mean:.3f}",
                    )
                    if _quit_pressed():
                        return
    finally:
        detector.close()
        camera_source.release()
        cv2.destroyAllWindows()


def run_occlusion_protocol(
    mode: str,
    camera: str,
    video: str | None,
    output: Path,
) -> None:
    config = AppConfig(source=parse_source(camera, video))
    max_faces = 1 if mode == "individual" else config.max_audience_faces
    camera_source = ResilientVideoSource(config.source, config.width, config.height)
    detector = create_detector(config, mode, max_faces)
    fields = ["timestamp", "seconds", "phase", "mode", "faces_detected"]
    phases = [("baseline", 5.0), ("oclusao", 3.0), ("recuperacao", 8.0)]
    start = time.monotonic()
    recovery_started_at: float | None = None
    recovered_at: float | None = None

    try:
        with CsvLogger(output, fields) as logger:
            for phase, duration in phases:
                phase_start = time.monotonic()
                if phase == "recuperacao":
                    recovery_started_at = phase_start
                while time.monotonic() - phase_start < duration:
                    now = time.monotonic()
                    ok, frame = camera_source.read()
                    faces_count = 0
                    if ok and frame is not None:
                        faces_count = len(detector.detect(frame))
                        if phase == "recuperacao" and faces_count > 0 and recovered_at is None:
                            recovered_at = now
                        _draw_protocol_text(frame, f"Oclusao: {phase} | q sai")
                        cv2.imshow("Protocolo Oclusao", frame)
                    else:
                        cv2.imshow("Protocolo Oclusao", blank_frame(config.width, config.height, "Fonte indisponivel. Tentando reconectar..."))
                    logger.write(
                        timestamp=timestamp_iso(),
                        seconds=f"{now - start:.3f}",
                        phase=phase,
                        mode=mode,
                        faces_detected=faces_count,
                    )
                    if _quit_pressed():
                        return
    finally:
        detector.close()
        camera_source.release()
        cv2.destroyAllWindows()

    if recovery_started_at and recovered_at:
        print(f"Tempo de recuperacao: {recovered_at - recovery_started_at:.2f}s")
    else:
        print("Recuperacao nao detectada no periodo observado.")


def run_failover_protocol(duration: float, camera: str, output: Path) -> None:
    config = AppConfig(source=parse_source(camera, None))
    camera_source = ResilientVideoSource(config.source, config.width, config.height)
    fields = ["timestamp", "seconds", "camera_available", "read_ok"]
    start = time.monotonic()

    try:
        with CsvLogger(output, fields) as logger:
            while time.monotonic() - start < duration:
                now = time.monotonic()
                ok, frame = camera_source.read()
                available = int(camera_source.capture is not None and camera_source.capture.isOpened())
                logger.write(
                    timestamp=timestamp_iso(),
                    seconds=f"{now - start:.3f}",
                    camera_available=available,
                    read_ok=int(ok),
                )
                if ok and frame is not None:
                    _draw_protocol_text(frame, "Failover: desconecte/reconecte a webcam | q sai")
                    cv2.imshow("Protocolo Failover", frame)
                else:
                    cv2.imshow("Protocolo Failover", blank_frame(config.width, config.height, "Webcam indisponivel. Tentando reconectar..."))
                if _quit_pressed():
                    break
    finally:
        camera_source.release()
        cv2.destroyAllWindows()


def create_detector(config: AppConfig, mode: str, max_faces: int) -> FaceLandmarkDetector:
    return FaceLandmarkDetector(
        max_faces=max_faces,
        refine_landmarks=(mode == "individual"),
        model_path=str(config.face_landmarker_model),
        detector_profile=config.detector_profile,
        yunet_model_path=str(config.yunet_model),
        scrfd_model_path=str(config.scrfd_model),
        scrfd_input_size=config.scrfd_input_size,
        min_detection_confidence=config.face_confidence,
    )


def evaluate_confusion_matrix(labels_csv: Path, output_dir: Path) -> None:
    truth: list[str] = []
    predicted: list[str] = []
    with labels_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("truth") and row.get("predicted"):
                truth.append(row["truth"].strip())
                predicted.append(row["predicted"].strip())

    if not truth:
        raise ValueError("CSV sem linhas validas. Use colunas truth,predicted.")

    labels = sorted(set(truth) | set(predicted))
    matrix = build_confusion_matrix(truth, predicted, labels)
    accuracy, precision, recall, f1 = classification_metrics(matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "confusion_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["samples", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
        writer.writeheader()
        writer.writerow(
            {
                "samples": len(truth),
                "accuracy": f"{accuracy:.5f}",
                "precision_macro": f"{precision:.5f}",
                "recall_macro": f"{recall:.5f}",
                "f1_macro": f"{f1:.5f}",
            }
        )

    figure_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(matrix, labels, figure_path, samples=len(truth))

    print(f"Amostras: {len(truth)}")
    print(f"Acuracia: {accuracy:.3f}")
    print(f"Precisao macro: {precision:.3f}")
    print(f"Recall macro: {recall:.3f}")
    print(f"F1 macro: {f1:.3f}")
    print(f"Arquivos: {metrics_path} | {figure_path}")


def build_confusion_matrix(truth: list[str], predicted: list[str], labels: list[str]) -> list[list[int]]:
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth_label, predicted_label in zip(truth, predicted):
        matrix[positions[truth_label]][positions[predicted_label]] += 1
    return matrix


def classification_metrics(matrix: list[list[int]]) -> tuple[float, float, float, float]:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(matrix)))
    accuracy = correct / total if total else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for index in range(len(matrix)):
        true_positive = matrix[index][index]
        predicted_positive = sum(row[index] for row in matrix)
        actual_positive = sum(matrix[index])
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    class_count = len(matrix) or 1
    return (
        accuracy,
        sum(precisions) / class_count,
        sum(recalls) / class_count,
        sum(f1_scores) / class_count,
    )


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], output: Path, samples: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusao - n={samples}")
    max_value = max((value for row in matrix for value in row), default=0)
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "#17202a"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _draw_protocol_text(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(frame, text, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (235, 235, 235), 2)


def _quit_pressed() -> bool:
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27)
