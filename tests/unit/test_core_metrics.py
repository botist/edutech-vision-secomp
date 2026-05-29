from __future__ import annotations

import numpy as np

from edutech_vision.core.aggregation import EngagementWindow
from edutech_vision.core.filters import MovingAverage, SustainedCondition
from edutech_vision.core.metrics import LEFT_EYE_EAR, MOUTH_MAR, RIGHT_EYE_EAR, mean_eye_aspect_ratio, mouth_aspect_ratio


def test_ear_and_mar_from_synthetic_landmarks() -> None:
    points = np.zeros((468, 2), dtype=np.float32)
    eye_points = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]
    for index, point in zip(LEFT_EYE_EAR, eye_points, strict=True):
        points[index] = point
    for index, point in zip(RIGHT_EYE_EAR, eye_points, strict=True):
        points[index] = point
    for index, point in zip(MOUTH_MAR, [(0, 0), (4, 0), (2, 1), (2, -1)], strict=True):
        points[index] = point

    assert mean_eye_aspect_ratio(points) == 0.5
    assert mouth_aspect_ratio(points) == 0.5


def test_moving_average_uses_sliding_window() -> None:
    avg = MovingAverage(maxlen=3)
    assert avg.update(10) == 10
    assert avg.update(20) == 15
    assert avg.update(30) == 20
    assert avg.update(40) == 30


def test_sustained_condition_only_fires_after_duration() -> None:
    condition = SustainedCondition(seconds=2.0)
    assert condition.update(True, now=10.0) is False
    assert condition.update(True, now=11.0) is False
    assert condition.update(True, now=12.1) is True
    assert condition.update(False, now=12.2) is False


def test_engagement_window_aggregates_last_seconds() -> None:
    window = EngagementWindow(seconds=10.0)
    assert window.update(0.0, faces=10, attentive=5) == 50.0
    assert window.update(5.0, faces=10, attentive=8) == 65.0
    assert window.update(12.0, faces=10, attentive=10) == 90.0
