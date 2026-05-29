from __future__ import annotations

import numpy as np


LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)
MOUTH_MAR = (61, 291, 13, 14)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, indices: tuple[int, int, int, int, int, int]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
    horizontal = euclidean(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    return vertical / (2.0 * horizontal)


def mean_eye_aspect_ratio(points: np.ndarray) -> float:
    left = eye_aspect_ratio(points, LEFT_EYE_EAR)
    right = eye_aspect_ratio(points, RIGHT_EYE_EAR)
    return (left + right) / 2.0


def mouth_aspect_ratio(points: np.ndarray) -> float:
    left_corner, right_corner, upper_lip, lower_lip = (points[index] for index in MOUTH_MAR)
    horizontal = euclidean(left_corner, right_corner)
    if horizontal <= 1e-6:
        return 0.0
    return euclidean(upper_lip, lower_lip) / horizontal
