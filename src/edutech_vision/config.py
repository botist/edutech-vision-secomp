from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
INDIVIDUAL_FACE_CONFIDENCE = 0.70
AUDIENCE_FACE_CONFIDENCE = 0.60


def default_face_confidence(mode: str) -> float:
    return INDIVIDUAL_FACE_CONFIDENCE if mode == "individual" else AUDIENCE_FACE_CONFIDENCE


@dataclass(frozen=True)
class AppConfig:
    source: int | str = 0
    width: int = 960
    height: int = 540
    face_landmarker_model: Path = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
    yunet_model: Path = ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    scrfd_model: Path = ROOT_DIR / "assets" / "models" / "scrfd_2.5g_bnkps.onnx"
    scrfd_input_size: int = 640
    detector_profile: str = "enhanced"
    face_confidence: float = AUDIENCE_FACE_CONFIDENCE
    mirror_camera: bool = True
    result_dir: Path = ROOT_DIR / "results"
    smoothing_window: int = 7
    calibration_seconds: float = 5.0
    fps_window_seconds: float = 60.0
    audience_window_seconds: float = 10.0
    max_audience_faces: int = 24
    stage_yaw_degrees: float = 0.0
    stage_pitch_degrees: float = 0.0
    audience_yaw_tolerance: float = 30.0
    audience_pitch_tolerance: float = 20.0
    posture_pitch_tolerance: float = 14.0
    attention_yaw_tolerance: float = 22.0
    eye_closed_seconds: float = 1.2
    yawn_seconds: float = 1.0
    posture_seconds: float = 2.0
    distraction_seconds: float = 2.0
    sound_enabled: bool = True
    showcase: bool = False
    fullscreen: bool = False


def parse_source(camera: str, video: str | None) -> int | str:
    if video:
        return video
    if camera.isdigit():
        return int(camera)
    return camera
