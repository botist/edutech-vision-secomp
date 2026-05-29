from __future__ import annotations

import cv2
import numpy as np

from edutech_vision.core.models import PoseResult


POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)

# Generic 3D face model in millimeters. The exact scale is less important than
# the relative geometry for estimating yaw/pitch/roll with solvePnP.
FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -63.6, -12.5),      # chin
        (-43.3, 32.7, -26.0),     # left eye outer corner
        (43.3, 32.7, -26.0),      # right eye outer corner
        (-28.9, -28.9, -24.1),    # left mouth corner
        (28.9, -28.9, -24.1),     # right mouth corner
    ],
    dtype=np.float64,
)


def estimate_head_pose(points: np.ndarray, frame_shape: tuple[int, int, int]) -> PoseResult:
    height, width = frame_shape[:2]
    image_points = np.array([points[index] for index in POSE_LANDMARKS], dtype=np.float64)

    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return PoseResult(success=False)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = (float(value) for value in euler.flatten()[:3])

    return PoseResult(
        success=True,
        yaw=_normalize_angle(yaw),
        pitch=_normalize_angle(pitch),
        roll=_normalize_angle(roll),
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
    )


def draw_head_pose_axis(
    frame: np.ndarray,
    points: np.ndarray,
    pose: PoseResult,
    length: float = 80.0,
) -> None:
    if not pose.success or pose.rotation_vector is None or pose.translation_vector is None:
        return

    height, width = frame.shape[:2]
    camera_matrix = np.array(
        [[float(width), 0.0, width / 2.0], [0.0, float(width), height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
    nose = tuple(points[1].astype(int))
    projected, _ = cv2.projectPoints(
        axis,
        pose.rotation_vector,
        pose.translation_vector,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    projected = projected.reshape(-1, 2).astype(int)
    cv2.line(frame, nose, tuple(projected[0]), (0, 0, 255), 2)
    cv2.line(frame, nose, tuple(projected[1]), (0, 255, 0), 2)
    cv2.line(frame, nose, tuple(projected[2]), (255, 0, 0), 2)


def _normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle
