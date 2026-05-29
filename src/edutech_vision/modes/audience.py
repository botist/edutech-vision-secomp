from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from edutech_vision.config import AppConfig
from edutech_vision.core.aggregation import EngagementWindow
from edutech_vision.core.filters import MovingAverage
from edutech_vision.core.landmarks import FaceLandmarkDetector
from edutech_vision.core.pose import PoseResult, estimate_head_pose
from edutech_vision.ui.renderer import BLUE, GREEN, RED, YELLOW, compose_dashboard, draw_anonymous_face, draw_calibration_meter, show_frame
from edutech_vision.utils.camera import ResilientVideoSource, blank_frame
from edutech_vision.utils.events import AlertTransitionTracker, EventTimeline
from edutech_vision.utils.fps import SlidingFPS
from edutech_vision.utils.logging import CsvLogger, timestamp_iso


class AudienceMode:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.engagement_smoother = MovingAverage(5)

    def run(self) -> None:
        camera = ResilientVideoSource(self.config.source, self.config.width, self.config.height)
        detector = FaceLandmarkDetector(
            max_faces=self.config.max_audience_faces,
            refine_landmarks=False,
            model_path=str(self.config.face_landmarker_model),
            detector_profile=self.config.detector_profile,
            yunet_model_path=str(self.config.yunet_model),
            scrfd_model_path=str(self.config.scrfd_model),
            scrfd_input_size=self.config.scrfd_input_size,
            min_detection_confidence=self.config.face_confidence,
        )
        fps = SlidingFPS(self.config.fps_window_seconds)
        detection_window = EngagementWindow(self.config.audience_window_seconds)
        pose_window = EngagementWindow(self.config.audience_window_seconds)
        calibrator = AudienceStageCalibrator(
            duration_seconds=self.config.calibration_seconds,
            default_yaw=self.config.stage_yaw_degrees,
            default_pitch=self.config.stage_pitch_degrees,
        )
        chart_values: deque[float] = deque(maxlen=120)
        event_timeline = EventTimeline()
        engagement_tracker = AlertTransitionTracker()
        start = time.monotonic()
        last_log = 0.0

        log_path = self.config.result_dir / "audience_engagement.csv"
        event_log_path = self.config.result_dir / "presentation_events.csv"
        fields = [
            "timestamp",
            "seconds",
            "window_seconds",
            "faces_mean",
            "faces_with_pose_mean",
            "attentive_mean",
            "engagement_percent",
            "fps_mean",
            "session_note",
        ]
        event_fields = ["timestamp", "seconds", "mode", "event", "state", "detail"]

        try:
            with CsvLogger(log_path, fields) as logger, CsvLogger(event_log_path, event_fields) as event_logger:
                while True:
                    now = time.monotonic()
                    frame_ok, frame = camera.read()
                    fps_stats = fps.update(now)

                    if not frame_ok or frame is None:
                        status = blank_frame(self.config.width, self.config.height, "Webcam indisponivel. Tentando reconectar...")
                        dashboard = compose_dashboard(
                            status,
                            "Modo Plateia",
                            [("Fonte", "desconectada")],
                            [("Recuperacao", False)],
                            f"FPS {fps_stats.current:.1f}",
                            events=event_timeline.formatted() if self.config.showcase else None,
                            insight="Failover ativo: a interface permanece viva enquanto tenta reconectar." if self.config.showcase else None,
                        )
                        show_frame("EduTech Vision", dashboard, fullscreen=self.config.fullscreen)
                        if _quit_pressed():
                            break
                        continue

                    if self.config.mirror_camera and isinstance(self.config.source, int):
                        frame = cv2.flip(frame, 1)

                    faces = detector.detect(frame)
                    attentive_count = 0
                    posed_count = 0
                    for face in faces:
                        if face.points is None:
                            draw_anonymous_face(frame, face, False)
                            continue
                        pose = estimate_head_pose(face.points, frame.shape)
                        if pose.success:
                            calibrator.update(pose, now)
                            posed_count += 1
                        attentive = self._is_attentive(pose, calibrator)
                        attentive_count += int(attentive)
                        draw_anonymous_face(frame, face, attentive)
                    if self.config.showcase and calibrator.active(now):
                        draw_calibration_meter(frame, now - calibrator.started_at, self.config.calibration_seconds)

                    instant_engagement = 0.0 if not posed_count else 100.0 * attentive_count / posed_count
                    smoothed_engagement = self.engagement_smoother.update(instant_engagement)
                    detection_window.update(now, len(faces), attentive_count)
                    window_engagement = pose_window.update(now, posed_count, attentive_count)
                    chart_values.append(window_engagement)

                    if now - last_log >= self.config.audience_window_seconds:
                        last_log = now
                        event = event_timeline.add(now - start, "Janela 10s", f"engajamento {window_engagement:.1f}%")
                        event_logger.write(
                            timestamp=timestamp_iso(),
                            seconds=f"{now - start:.3f}",
                            mode="plateia",
                            event=event.label,
                            state="sample",
                            detail=event.detail,
                        )
                        for event_name, active in engagement_tracker.update(
                            {
                                "Engajamento alto": window_engagement >= 70.0,
                                "Engajamento baixo": posed_count > 0 and window_engagement < 40.0,
                            }
                        ):
                            transition = event_timeline.add(now - start, event_name, "acionado" if active else "normalizado")
                            event_logger.write(
                                timestamp=timestamp_iso(),
                                seconds=f"{now - start:.3f}",
                                mode="plateia",
                                event=transition.label,
                                state="on" if active else "off",
                                detail=transition.detail,
                            )
                        logger.write(
                            timestamp=timestamp_iso(),
                            seconds=f"{now - start:.3f}",
                            window_seconds=f"{self.config.audience_window_seconds:.1f}",
                            faces_mean=f"{detection_window.mean_faces:.3f}",
                            faces_with_pose_mean=f"{pose_window.mean_faces:.3f}",
                            attentive_mean=f"{pose_window.mean_attentive:.3f}",
                            engagement_percent=f"{window_engagement:.3f}",
                            fps_mean=f"{fps_stats.mean:.3f}",
                            session_note="janela 10s pronta para graficos",
                        )

                    metrics = [
                        ("Faces", str(len(faces))),
                        ("Com pose valida", str(posed_count)),
                        ("Atentos", str(attentive_count)),
                        ("Engaj. inst.", f"{smoothed_engagement:.1f}%"),
                        ("Janela 10s", f"{window_engagement:.1f}%"),
                        ("Eixo palco", f"Y {calibrator.yaw:+.1f} / P {calibrator.pitch:+.1f}"),
                    ]
                    statuses = [
                        ("Metricas agregadas", True),
                        ("Telemetria 10s", True),
                        ("Calibrando", calibrator.active(now)),
                        ("Plateia atenta", window_engagement >= 60.0),
                    ]
                    bars = [
                        ("Engajamento", window_engagement, 0.0, 100.0, GREEN if window_engagement >= 60.0 else YELLOW),
                        ("Faces", float(len(faces)), 0.0, float(max(1, self.config.max_audience_faces)), BLUE),
                        ("Atentos", float(attentive_count), 0.0, float(max(1, self.config.max_audience_faces)), GREEN),
                        ("Nao atentos", float(max(0, posed_count - attentive_count)), 0.0, float(max(1, self.config.max_audience_faces)), RED),
                        ("Sem pose", float(max(0, len(faces) - posed_count)), 0.0, float(max(1, self.config.max_audience_faces)), YELLOW),
                    ]
                    dashboard = compose_dashboard(
                        frame,
                        "Modo Plateia",
                        metrics,
                        statuses,
                        f"FPS {fps_stats.current:.1f} | media {fps_stats.mean:.1f}",
                        list(chart_values),
                        bars=bars if self.config.showcase else None,
                        events=event_timeline.formatted() if self.config.showcase else None,
                        insight=f"Pipeline ao vivo: {self.config.detector_profile} -> pose valida -> engajamento 10s -> relatorio."
                        if self.config.showcase
                        else None,
                    )
                    show_frame("EduTech Vision", dashboard, fullscreen=self.config.fullscreen)
                    if _quit_pressed():
                        break
        finally:
            detector.close()
            camera.release()
            cv2.destroyAllWindows()

    def _is_attentive(self, pose: PoseResult, calibrator: "AudienceStageCalibrator") -> bool:
        if not pose.success:
            return False
        yaw_ok = abs(pose.yaw - calibrator.yaw) <= self.config.audience_yaw_tolerance
        pitch_ok = abs(pose.pitch - calibrator.pitch) <= self.config.audience_pitch_tolerance
        return yaw_ok and pitch_ok

def _quit_pressed() -> bool:
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27)


@dataclass
class AudienceStageCalibrator:
    duration_seconds: float
    default_yaw: float
    default_pitch: float

    def __post_init__(self) -> None:
        self.started_at = time.monotonic()
        self.yaw_samples: list[float] = []
        self.pitch_samples: list[float] = []

    def active(self, now: float) -> bool:
        return now - self.started_at < self.duration_seconds

    def update(self, pose: PoseResult, now: float) -> None:
        if not self.active(now):
            return
        self.yaw_samples.append(pose.yaw)
        self.pitch_samples.append(pose.pitch)

    @property
    def yaw(self) -> float:
        if len(self.yaw_samples) < 5:
            return self.default_yaw
        return float(np.median(np.array(self.yaw_samples, dtype=np.float32)))

    @property
    def pitch(self) -> float:
        if len(self.pitch_samples) < 5:
            return self.default_pitch
        return float(np.median(np.array(self.pitch_samples, dtype=np.float32)))
