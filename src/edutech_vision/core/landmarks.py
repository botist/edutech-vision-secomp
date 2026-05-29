from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from edutech_vision.config import ROOT_DIR
from edutech_vision.core.detection import ScrfdFaceDetector, YuNetFaceDetector, expanded_crop
from edutech_vision.core.models import FaceLandmarks


DEFAULT_TASK_MODEL = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
DEFAULT_YUNET_MODEL = ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_SCRFD_MODEL = ROOT_DIR / "assets" / "models" / "scrfd_2.5g_bnkps.onnx"
DETECTOR_PROFILES = ("mediapipe", "enhanced", "research")


class FaceLandmarkDetector:
    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.55,
        model_path: str | None = None,
        detector_profile: str = "enhanced",
        yunet_model_path: str | None = None,
        scrfd_model_path: str | None = None,
        scrfd_input_size: int = 640,
        running_mode: str = "video",
    ) -> None:
        if detector_profile not in DETECTOR_PROFILES:
            raise ValueError(f"Perfil de detector invalido: {detector_profile}")
        self.max_faces = max_faces
        self.detector_profile = detector_profile
        self.min_landmark_face_pixels = 0 if max_faces == 1 else 48
        self._direct = _MediaPipeLandmarker(
            max_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
            running_mode=running_mode,
        )
        self._roi_landmarker: _MediaPipeLandmarker | None = None
        self._box_detector: YuNetFaceDetector | ScrfdFaceDetector | None = None
        if detector_profile == "mediapipe":
            return

        self._roi_landmarker = _MediaPipeLandmarker(
            max_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
            running_mode="image",
        )
        if detector_profile == "enhanced":
            self._box_detector = YuNetFaceDetector(
                Path(yunet_model_path) if yunet_model_path else DEFAULT_YUNET_MODEL,
                confidence=min_detection_confidence,
                max_faces=max_faces,
                multiscale=max_faces == 1,
            )
        else:
            self._box_detector = ScrfdFaceDetector(
                Path(scrfd_model_path) if scrfd_model_path else DEFAULT_SCRFD_MODEL,
                confidence=min_detection_confidence,
                max_faces=max_faces,
                input_size=scrfd_input_size,
            )

    def detect(self, frame_bgr: np.ndarray) -> list[FaceLandmarks]:
        if self._box_detector is None or self._roi_landmarker is None:
            return self._direct.detect(frame_bgr)
        detected = []
        for candidate in self._box_detector.detect(frame_bgr):
            x, y, width, height = candidate.bbox
            if min(width, height) < self.min_landmark_face_pixels:
                if _border_clipped(candidate.bbox, frame_bgr.shape):
                    continue
                detected.append(
                    FaceLandmarks(
                        points=None,
                        visibility=candidate.score,
                        detected_bbox=(int(x), int(y), int(width), int(height)),
                    )
                )
                continue
            crop_data = expanded_crop(frame_bgr, candidate.bbox)
            if crop_data is None:
                continue
            crop, (x_offset, y_offset) = crop_data
            crop_faces = self._roi_landmarker.detect(crop)
            if not crop_faces:
                if _border_clipped(candidate.bbox, frame_bgr.shape):
                    continue
                detected.append(
                    FaceLandmarks(
                        points=None,
                        visibility=candidate.score,
                        detected_bbox=(int(x), int(y), int(width), int(height)),
                    )
                )
                continue
            face = max(crop_faces, key=lambda item: item.bbox[2] * item.bbox[3])
            if face.points is None:
                continue
            mapped = face.points.copy()
            mapped[:, 0] += x_offset
            mapped[:, 1] += y_offset
            detected.append(
                FaceLandmarks(
                    points=mapped,
                    visibility=candidate.score,
                    detected_bbox=(int(x), int(y), int(width), int(height)),
                )
            )
        return detected

    def close(self) -> None:
        self._direct.close()
        if self._roi_landmarker is not None:
            self._roi_landmarker.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _border_clipped(bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> bool:
    height, width = frame_shape[:2]
    x, y, box_width, box_height = bbox
    return x < 0 or y < 0 or x + box_width > width or y + box_height > height


class _MediaPipeLandmarker:
    def __init__(
        self,
        max_faces: int,
        refine_landmarks: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: str | None,
        running_mode: str,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._timestamp_ms = 0
        self._running_mode = running_mode
        self._backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._face_mesh = None
        self._landmarker = None

        if self._backend == "solutions":
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=running_mode == "image",
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        task_model = DEFAULT_TASK_MODEL if model_path is None else Path(model_path)
        if not task_model.is_absolute():
            task_model = DEFAULT_TASK_MODEL.parent / task_model
        if not task_model.exists():
            raise FileNotFoundError(f"Modelo MediaPipe nao encontrado em {task_model}. Rode run.bat ou run.sh.")

        mode = VisionTaskRunningMode.IMAGE if running_mode == "image" else VisionTaskRunningMode.VIDEO
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_model)),
            running_mode=mode,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceLandmarks]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            return self._detect_solutions(frame_rgb, width, height)
        return self._detect_tasks(frame_rgb, width, height)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def _detect_solutions(self, frame_rgb: np.ndarray, width: int, height: int) -> list[FaceLandmarks]:
        if self._face_mesh is None:
            return []
        frame_rgb.flags.writeable = False
        result = self._face_mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return []
        return [
            FaceLandmarks(
                points=np.array(
                    [(landmark.x * width, landmark.y * height) for landmark in face.landmark],
                    dtype=np.float32,
                )
            )
            for face in result.multi_face_landmarks
        ]

    def _detect_tasks(self, frame_rgb: np.ndarray, width: int, height: int) -> list[FaceLandmarks]:
        if self._landmarker is None:
            return []
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        if self._running_mode == "image":
            result = self._landmarker.detect(image)
        else:
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.face_landmarks:
            return []
        return [
            FaceLandmarks(
                points=np.array(
                    [(landmark.x * width, landmark.y * height) for landmark in face],
                    dtype=np.float32,
                )
            )
            for face in result.face_landmarks
        ]
