from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceLandmarks:
    points: np.ndarray | None
    visibility: float = 1.0
    detected_bbox: tuple[int, int, int, int] | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        if self.detected_bbox is not None:
            return self.detected_bbox
        if self.points is None:
            return (0, 0, 0, 0)
        x, y, w, h = cv2_bounding_rect(self.points)
        return int(x), int(y), int(w), int(h)


@dataclass(frozen=True)
class PoseResult:
    success: bool
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    rotation_vector: np.ndarray | None = None
    translation_vector: np.ndarray | None = None


@dataclass(frozen=True)
class IndividualMetrics:
    ear: float
    mar: float
    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class AudienceFrameSummary:
    faces_detected: int
    attentive_count: int
    engagement_percent: float


def cv2_bounding_rect(points: np.ndarray) -> tuple[int, int, int, int]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    x, y = min_xy
    w, h = max_xy - min_xy
    return int(x), int(y), int(w), int(h)
