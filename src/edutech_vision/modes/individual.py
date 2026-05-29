from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from edutech_vision.config import AppConfig
from edutech_vision.core.filters import MetricSmoother, SustainedCondition
from edutech_vision.core.landmarks import FaceLandmarkDetector
from edutech_vision.core.metrics import mean_eye_aspect_ratio, mouth_aspect_ratio
from edutech_vision.core.models import IndividualMetrics
from edutech_vision.core.pose import draw_head_pose_axis, estimate_head_pose
from edutech_vision.ui.renderer import BLUE, GREEN, RED, YELLOW, compose_dashboard, draw_calibration_meter, draw_face_box, show_frame
from edutech_vision.utils.audio import AlertSound
from edutech_vision.utils.camera import ResilientVideoSource, blank_frame
from edutech_vision.utils.events import AlertTransitionTracker, EventTimeline
from edutech_vision.utils.fps import SlidingFPS
from edutech_vision.utils.logging import CsvLogger, timestamp_iso


@dataclass(frozen=True)
class Baseline:
    ear: float = 0.28
    mar: float = 0.05
    yaw: float = 0.0
    pitch: float = 0.0


class BaselineCalibrator:
    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.started_at = time.monotonic()
        self.samples: list[IndividualMetrics] = []

    def update(self, metrics: IndividualMetrics, now: float) -> None:
        if self.active(now):
            self.samples.append(metrics)

    def active(self, now: float) -> bool:
        return now - self.started_at < self.duration_seconds

    @property
    def baseline(self) -> Baseline:
        if len(self.samples) < 10:
            return Baseline()
        ears = np.array([sample.ear for sample in self.samples], dtype=np.float32)
        mars = np.array([sample.mar for sample in self.samples], dtype=np.float32)
        yaws = np.array([sample.yaw for sample in self.samples], dtype=np.float32)
        pitches = np.array([sample.pitch for sample in self.samples], dtype=np.float32)
        return Baseline(
            ear=float(np.median(ears)),
            mar=float(np.median(mars)),
            yaw=float(np.median(yaws)),
            pitch=float(np.median(pitches)),
        )


class IndividualMode:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.smoother = MetricSmoother(config.smoothing_window)
        self.eye_closed = SustainedCondition(config.eye_closed_seconds)
        self.yawn = SustainedCondition(config.yawn_seconds)
        self.bad_posture = SustainedCondition(config.posture_seconds)
        self.distraction = SustainedCondition(config.distraction_seconds)
        self.sound = AlertSound(enabled=config.sound_enabled)

    def run(self) -> None:
        camera = ResilientVideoSource(self.config.source, self.config.width, self.config.height)
        detector = FaceLandmarkDetector(
            max_faces=1,
            model_path=str(self.config.face_landmarker_model),
            detector_profile=self.config.detector_profile,
            yunet_model_path=str(self.config.yunet_model),
            scrfd_model_path=str(self.config.scrfd_model),
            scrfd_input_size=self.config.scrfd_input_size,
            min_detection_confidence=self.config.face_confidence,
        )
        fps = SlidingFPS(self.config.fps_window_seconds)
        calibrator = BaselineCalibrator(self.config.calibration_seconds)
        event_timeline = EventTimeline()
        alert_tracker = AlertTransitionTracker()
        start = time.monotonic()

        log_path = self.config.result_dir / "individual_session.csv"
        event_log_path = self.config.result_dir / "presentation_events.csv"
        fields = [
            "timestamp",
            "seconds",
            "face_detected",
            "ear",
            "mar",
            "yaw",
            "pitch",
            "roll",
            "eye_closed_alert",
            "yawn_alert",
            "posture_alert",
            "distraction_alert",
            "fatigue_alert",
            "fps_current",
            "fps_mean",
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
                            "Modo Individual",
                            [("Fonte", "desconectada")],
                            [("Recuperacao", False)],
                            f"FPS {fps_stats.current:.1f}",
                            events=event_timeline.formatted() if self.config.showcase else None,
                            insight="Failover ativo: o sistema tenta reconectar sem encerrar a demonstracao." if self.config.showcase else None,
                        )
                        show_frame("EduTech Vision", dashboard, fullscreen=self.config.fullscreen)
                        if _quit_pressed():
                            break
                        continue

                    if self.config.mirror_camera and isinstance(self.config.source, int):
                        frame = cv2.flip(frame, 1)

                    faces = detector.detect(frame)
                    metrics_rows: list[tuple[str, str]] = []
                    statuses: list[tuple[str, bool]] = []
                    bars: list[tuple[str, float, float, float, tuple[int, int, int]]] = []

                    if faces and faces[0].points is not None:
                        face = faces[0]
                        raw = self._compute_metrics(face.points, frame)
                        calibrator.update(raw, now)
                        smoothed_values = self.smoother.update(
                            ear=raw.ear,
                            mar=raw.mar,
                            yaw=raw.yaw,
                            pitch=raw.pitch,
                            roll=raw.roll,
                        )
                        metrics = IndividualMetrics(**smoothed_values)
                        baseline = calibrator.baseline
                        thresholds = self._thresholds(baseline)

                        eye_alert = self.eye_closed.update(metrics.ear < thresholds["ear"], now)
                        yawn_alert = self.yawn.update(metrics.mar > thresholds["mar"], now)
                        posture_alert = self.bad_posture.update(
                            abs(metrics.pitch - baseline.pitch) > self.config.posture_pitch_tolerance,
                            now,
                        )
                        distraction_alert = self.distraction.update(
                            abs(metrics.yaw - baseline.yaw) > self.config.attention_yaw_tolerance,
                            now,
                        )
                        fatigue_alert = eye_alert or yawn_alert

                        if fatigue_alert or posture_alert or distraction_alert:
                            self.sound.play()

                        states = {
                            "Rastreamento": True,
                            "Fadiga": fatigue_alert,
                            "Bocejo": yawn_alert,
                            "Postura": posture_alert,
                            "Desatencao": distraction_alert,
                        }
                        for event_name, active in alert_tracker.update(states):
                            detail = "acionado" if active else "normalizado"
                            event = event_timeline.add(now - start, event_name, detail)
                            event_logger.write(
                                timestamp=timestamp_iso(),
                                seconds=f"{now - start:.3f}",
                                mode="individual",
                                event=event_name,
                                state="on" if active else "off",
                                detail=event.detail,
                            )

                        draw_face_box(frame, face)
                        pose = estimate_head_pose(face.points, frame.shape)
                        draw_head_pose_axis(frame, face.points, pose)
                        self._draw_alert_banner(frame, fatigue_alert, posture_alert, distraction_alert, calibrator.active(now))
                        if self.config.showcase and calibrator.active(now):
                            draw_calibration_meter(frame, now - calibrator.started_at, self.config.calibration_seconds)

                        yaw_delta = metrics.yaw - baseline.yaw
                        pitch_delta = metrics.pitch - baseline.pitch
                        metrics_rows = [
                            ("EAR", f"{metrics.ear:.3f} (lim {thresholds['ear']:.3f})"),
                            ("MAR", f"{metrics.mar:.3f} (lim {thresholds['mar']:.3f})"),
                            ("Yaw delta", f"{yaw_delta:+.1f} deg"),
                            ("Pitch delta", f"{pitch_delta:+.1f} deg"),
                        ]
                        statuses = [
                            ("Fadiga", fatigue_alert),
                            ("Bocejo", yawn_alert),
                            ("Postura ruim", posture_alert),
                            ("Desatencao", distraction_alert),
                        ]
                        bars = [
                            ("EAR olho", metrics.ear, 0.0, 0.38, GREEN if not eye_alert else RED),
                            ("MAR boca", metrics.mar, 0.0, 0.45, YELLOW if yawn_alert else GREEN),
                            ("Yaw delta", abs(yaw_delta), 0.0, self.config.attention_yaw_tolerance * 1.5, RED if distraction_alert else BLUE),
                            ("Pitch delta", abs(pitch_delta), 0.0, self.config.posture_pitch_tolerance * 1.5, RED if posture_alert else BLUE),
                        ]

                        logger.write(
                            timestamp=timestamp_iso(),
                            seconds=f"{now - start:.3f}",
                            face_detected=1,
                            ear=f"{metrics.ear:.5f}",
                            mar=f"{metrics.mar:.5f}",
                            yaw=f"{metrics.yaw:.3f}",
                            pitch=f"{metrics.pitch:.3f}",
                            roll=f"{metrics.roll:.3f}",
                            eye_closed_alert=int(eye_alert),
                            yawn_alert=int(yawn_alert),
                            posture_alert=int(posture_alert),
                            distraction_alert=int(distraction_alert),
                            fatigue_alert=int(fatigue_alert),
                            fps_current=f"{fps_stats.current:.3f}",
                            fps_mean=f"{fps_stats.mean:.3f}",
                        )
                    elif faces:
                        face = faces[0]
                        self.eye_closed.reset()
                        self.yawn.reset()
                        self.bad_posture.reset()
                        self.distraction.reset()
                        draw_face_box(frame, face, YELLOW)
                        metrics_rows = [("Face", "detectada"), ("Landmarks", "insuficientes para metricas")]
                        statuses = [("Rastreamento", True), ("Landmarks", False)]
                        for event_name, active in alert_tracker.update(
                            {
                                "Rastreamento": True,
                                "Fadiga": False,
                                "Bocejo": False,
                                "Postura": False,
                                "Desatencao": False,
                            }
                        ):
                            detail = "acionado" if active else "normalizado"
                            event = event_timeline.add(now - start, event_name, detail)
                            event_logger.write(
                                timestamp=timestamp_iso(),
                                seconds=f"{now - start:.3f}",
                                mode="individual",
                                event=event_name,
                                state="on" if active else "off",
                                detail=event.detail,
                            )
                        logger.write(
                            timestamp=timestamp_iso(),
                            seconds=f"{now - start:.3f}",
                            face_detected=1,
                            fps_current=f"{fps_stats.current:.3f}",
                            fps_mean=f"{fps_stats.mean:.3f}",
                        )
                    else:
                        self.eye_closed.reset()
                        self.yawn.reset()
                        self.bad_posture.reset()
                        self.distraction.reset()
                        metrics_rows = [("Face", "nao detectada")]
                        statuses = [("Rastreamento", False)]
                        states = {
                            "Rastreamento": False,
                            "Fadiga": False,
                            "Bocejo": False,
                            "Postura": False,
                            "Desatencao": False,
                        }
                        for event_name, active in alert_tracker.update(states):
                            if not active:
                                event = event_timeline.add(now - start, event_name, "normalizado" if event_name != "Rastreamento" else "face ausente")
                                event_logger.write(
                                    timestamp=timestamp_iso(),
                                    seconds=f"{now - start:.3f}",
                                    mode="individual",
                                    event=event_name,
                                    state="off",
                                    detail=event.detail,
                                )
                        logger.write(
                            timestamp=timestamp_iso(),
                            seconds=f"{now - start:.3f}",
                            face_detected=0,
                            fps_current=f"{fps_stats.current:.3f}",
                            fps_mean=f"{fps_stats.mean:.3f}",
                        )

                    dashboard = compose_dashboard(
                        frame,
                        "Modo Individual",
                        metrics_rows,
                        statuses,
                        f"FPS {fps_stats.current:.1f} | media {fps_stats.mean:.1f}",
                        bars=bars if self.config.showcase else None,
                        events=event_timeline.formatted() if self.config.showcase else None,
                        insight=f"Ao vivo: {self.config.detector_profile} -> landmarks -> EAR/MAR -> solvePnP -> alerta sustentado."
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

    @staticmethod
    def _compute_metrics(points: np.ndarray, frame: np.ndarray) -> IndividualMetrics:
        pose = estimate_head_pose(points, frame.shape)
        return IndividualMetrics(
            ear=mean_eye_aspect_ratio(points),
            mar=mouth_aspect_ratio(points),
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
        )

    @staticmethod
    def _thresholds(baseline: Baseline) -> dict[str, float]:
        ear_threshold = max(0.15, min(0.24, baseline.ear * 0.72))
        mar_threshold = max(0.18, baseline.mar * 2.4)
        return {"ear": ear_threshold, "mar": mar_threshold}

    @staticmethod
    def _draw_alert_banner(
        frame: np.ndarray,
        fatigue_alert: bool,
        posture_alert: bool,
        distraction_alert: bool,
        calibrating: bool,
    ) -> None:
        messages: list[str] = []
        if calibrating:
            messages.append("Calibrando baseline neutro...")
        if fatigue_alert:
            messages.append("ALERTA: fadiga sustentada")
        if posture_alert:
            messages.append("ALERTA: queda postural")
        if distraction_alert:
            messages.append("ALERTA: desatencao")
        if not messages:
            return
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(frame, " | ".join(messages), (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 230, 255), 2)


def _quit_pressed() -> bool:
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27)
