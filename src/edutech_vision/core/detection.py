from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    score: float
    source: str
    keypoints: np.ndarray | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2]) * max(0.0, self.bbox[3])


def intersection_over_union(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union else 0.0


def non_max_suppression(detections: list[FaceDetection], threshold: float = 0.35) -> list[FaceDetection]:
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    selected: list[FaceDetection] = []
    for candidate in ordered:
        if all(intersection_over_union(candidate.bbox, prior.bbox) < threshold for prior in selected):
            selected.append(candidate)
    return selected


def expanded_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: float = 0.28,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    height, width = frame.shape[:2]
    x, y, box_width, box_height = bbox
    side_pad = box_width * padding
    vertical_pad = box_height * padding
    x1 = max(0, int(round(x - side_pad)))
    y1 = max(0, int(round(y - vertical_pad)))
    x2 = min(width, int(round(x + box_width + side_pad)))
    y2 = min(height, int(round(y + box_height + vertical_pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2], (x1, y1)


def prioritize_detections(
    detections: list[FaceDetection],
    frame_shape: tuple[int, ...],
    max_faces: int,
) -> list[FaceDetection]:
    height, width = frame_shape[:2]
    if max_faces == 1:
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

        def individual_rank(item: FaceDetection) -> float:
            x, y, box_width, box_height = item.bbox
            face_center = np.array([x + box_width / 2.0, y + box_height / 2.0], dtype=np.float32)
            normalized_distance = float(np.linalg.norm(face_center - center)) / max(width, height)
            return item.area * (1.0 - min(0.55, normalized_distance)) * (0.75 + 0.25 * item.score)

        return sorted(detections, key=individual_rank, reverse=True)[:1]
    return sorted(detections, key=lambda item: (item.score, item.area), reverse=True)[:max_faces]


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        max_faces: int,
        multiscale: bool = False,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo YuNet nao encontrado em {model_path}. Rode run.bat ou run.sh.")
        self.max_faces = max_faces
        self.multiscale = multiscale
        self.detector = cv2.FaceDetectorYN_create(str(model_path), "", (320, 320), confidence, 0.3, 5000)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        scales = (1.0, 1.5) if self.multiscale and self.max_faces > 1 else (1.0,)
        detections = self._detect_scales(frame, scales)
        if self.multiscale and self.max_faces == 1 and not detections:
            detections = self._detect_scales(frame, (1.5,))
        return prioritize_detections(non_max_suppression(detections), frame.shape, self.max_faces)

    def _detect_scales(self, frame: np.ndarray, scales: tuple[float, ...]) -> list[FaceDetection]:
        results: list[FaceDetection] = []
        for scale in scales:
            candidate = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            height, width = candidate.shape[:2]
            self.detector.setInputSize((width, height))
            _, raw = self.detector.detect(candidate)
            if raw is None:
                continue
            for row in raw:
                x, y, box_width, box_height = (float(value) / scale for value in row[:4])
                keypoints = row[4:14].reshape(5, 2).astype(np.float32) / scale
                results.append(
                    FaceDetection(
                        bbox=(x, y, box_width, box_height),
                        score=float(row[14]),
                        source="yunet",
                        keypoints=keypoints,
                    )
                )
        return results


class ScrfdFaceDetector:
    """ONNX SCRFD adapter based on the official InsightFace output layout."""

    def __init__(self, model_path: Path, confidence: float, max_faces: int, input_size: int = 640) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo SCRFD nao encontrado em {model_path}. "
                "Rode scripts/download_models.py --research pelo Python da .venv."
            )
        if input_size < 320 or input_size % 32 != 0:
            raise ValueError("SCRFD input_size deve ser multiplo de 32 e >= 320.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime ausente. Rode run.bat ou run.sh para reparar o setup.") from exc

        self.confidence = confidence
        self.max_faces = max_faces
        self.input_size = (input_size, input_size)
        self.strides = (8, 16, 32)
        self.num_anchors = 2
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        image, scale = self._prepare(frame)
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})
        detections: list[FaceDetection] = []
        levels = len(self.strides)
        for index, stride in enumerate(self.strides):
            scores = outputs[index].reshape(-1)
            bbox_predictions = outputs[index + levels].reshape(-1, 4) * stride
            keypoint_predictions = outputs[index + levels * 2].reshape(-1, 10) * stride
            anchors = self._anchors(stride)
            selected = np.where(scores >= self.confidence)[0]
            if not selected.size:
                continue
            boxes = _distance_to_boxes(anchors, bbox_predictions)[selected] / scale
            keypoints = _distance_to_keypoints(anchors, keypoint_predictions)[selected] / scale
            for box, landmarks, score in zip(boxes, keypoints, scores[selected], strict=True):
                x1, y1, x2, y2 = box
                detections.append(
                    FaceDetection(
                        bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        score=float(score),
                        source="scrfd",
                        keypoints=landmarks,
                    )
                )
        return prioritize_detections(non_max_suppression(detections, threshold=0.4), frame.shape, self.max_faces)

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        input_width, input_height = self.input_size
        height, width = frame.shape[:2]
        scale = min(input_width / width, input_height / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas, scale

    def _anchors(self, stride: int) -> np.ndarray:
        height = self.input_size[1] // stride
        width = self.input_size[0] // stride
        grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32) * stride
        return np.stack([grid] * self.num_anchors, axis=2).reshape((-1, 2))


def _distance_to_boxes(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def _distance_to_keypoints(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    keypoints: list[np.ndarray] = []
    for index in range(0, distance.shape[1], 2):
        keypoints.append(points[:, 0] + distance[:, index])
        keypoints.append(points[:, 1] + distance[:, index + 1])
    return np.stack(keypoints, axis=-1).reshape((-1, 5, 2))
