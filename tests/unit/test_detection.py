from __future__ import annotations

import numpy as np

from edutech_vision.core.detection import (
    FaceDetection,
    expanded_crop,
    intersection_over_union,
    non_max_suppression,
    prioritize_detections,
)


def test_intersection_over_union_and_nms_remove_duplicate_faces() -> None:
    first = FaceDetection((10, 10, 40, 40), 0.95, "test")
    duplicate = FaceDetection((12, 12, 40, 40), 0.80, "test")
    separate = FaceDetection((100, 100, 20, 20), 0.75, "test")

    assert intersection_over_union(first.bbox, duplicate.bbox) > 0.8
    assert non_max_suppression([first, duplicate, separate]) == [first, separate]


def test_expanded_crop_stays_inside_frame() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    result = expanded_crop(frame, (0, 0, 30, 40), padding=0.5)

    assert result is not None
    crop, offset = result
    assert offset == (0, 0)
    assert crop.shape[0] == 60
    assert crop.shape[1] == 45


def test_individual_prioritizes_large_central_face() -> None:
    central = FaceDetection((35, 25, 35, 35), 0.85, "test")
    edge = FaceDetection((0, 0, 38, 38), 0.95, "test")

    assert prioritize_detections([edge, central], (100, 100, 3), max_faces=1) == [central]
