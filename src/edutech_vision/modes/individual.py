from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from edutech_vision.config import AppConfig
from edutech_vision.core.filters import MetricSmoother, SustainedCondition
from edutech_vision.core.landmarks import FaceLandmarkDetector
from edutech_vision.core.metrics import LEFT_EYE_EAR, RIGHT_EYE_EAR, eye_aspect_ratio, mouth_aspect_ratio
from edutech_vision.core.models import IndividualMetrics
from edutech_vision.core.pose import PoseResult, draw_head_pose_axis, estimate_head_pose
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
    ear_mad: float = 0.0
    mar_mad: float = 0.0
    yaw_mad: float = 0.0
    pitch_mad: float = 0.0
    samples: int = 0

    @property
    def calibrated(self) -> bool:
        return self.samples >= 10


@dataclass(frozen=True)
class IndividualAnalysis:
    metrics: IndividualMetrics
    pose: PoseResult
    left_ear: float
    right_ear: float
    quality_ok: bool
    quality_note: str


def _median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _angle_delta(value: float, reference: float) -> float:
    delta = value - reference
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def _circular_mean_degrees(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    radians = np.deg2rad(values.astype(np.float64))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) < 1e-9 and abs(cos_mean) < 1e-9:
        return float(np.median(values))
    return _angle_delta(float(np.rad2deg(np.arctan2(sin_mean, cos_mean))), 0.0)


def _angle_mad(values: np.ndarray, center: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median([abs(_angle_delta(float(value), center)) for value in values]))


def _frontal_pitch_delta(pitch: float) -> float:
    candidates = (_angle_delta(pitch, 0.0), _angle_delta(pitch, 180.0), _angle_delta(pitch, -180.0))
    return min(candidates, key=abs)


class BaselineCalibrator:
    min_samples = 10

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.started_at = time.monotonic()
        self.first_sample_at: float | None = None
        self.samples: list[IndividualMetrics] = []

    def update(self, metrics: IndividualMetrics, now: float) -> None:
        if self.active(now) and self._is_baseline_sample(metrics):
            if self.first_sample_at is None:
                self.first_sample_at = now
            self.samples.append(metrics)

    def active(self, now: float) -> bool:
        if self.first_sample_at is None:
            return True
        return len(self.samples) < self.min_samples or now - self.first_sample_at < self.duration_seconds

    def elapsed(self, now: float) -> float:
        if self.first_sample_at is None:
            return 0.0
        return max(0.0, now - self.first_sample_at)

    @property
    def baseline(self) -> Baseline:
        if len(self.samples) < self.min_samples:
            return Baseline()
        ears = np.array([sample.ear for sample in self.samples], dtype=np.float32)
        mars = np.array([sample.mar for sample in self.samples], dtype=np.float32)
        yaws = np.array([sample.yaw for sample in self.samples], dtype=np.float32)
        pitches = np.array([sample.pitch for sample in self.samples], dtype=np.float32)
        yaw = _circular_mean_degrees(yaws)
        pitch = _circular_mean_degrees(pitches)
        return Baseline(
            ear=float(np.median(ears)),
            mar=float(np.median(mars)),
            yaw=yaw,
            pitch=pitch,
            ear_mad=_median_absolute_deviation(ears),
            mar_mad=_median_absolute_deviation(mars),
            yaw_mad=_angle_mad(yaws, yaw),
            pitch_mad=_angle_mad(pitches, pitch),
            samples=len(self.samples),
        )

    @staticmethod
    def _is_baseline_sample(metrics: IndividualMetrics) -> bool:
        values = np.array([metrics.ear, metrics.mar, metrics.yaw, metrics.pitch, metrics.roll], dtype=np.float32)
        if not np.isfinite(values).all():
            return False
        return (
            0.16 <= metrics.ear <= 0.45
            and 0.0 <= metrics.mar <= 0.24
            and abs(metrics.yaw) <= 70.0
            and abs(_frontal_pitch_delta(metrics.pitch)) <= 70.0
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
            "tracking_quality",
            "quality_note",
            "baseline_samples",
            "baseline_calibrated",
            "ear",
            "left_ear",
            "right_ear",
            "mar",
            "yaw",
            "pitch",
            "roll",
            "ear_threshold",
            "mar_threshold",
            "yaw_delta",
            "pitch_delta",
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
                        self._reset_tracking_state()
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
                        analysis = self._compute_analysis(face.points, frame)
                        baseline = calibrator.baseline
                        thresholds = self._thresholds(baseline)

                        if not analysis.quality_ok:
                            self._reset_tracking_state()
                            self._write_state_transitions(
                                alert_tracker,
                                event_timeline,
                                event_logger,
                                start,
                                now,
                                {
                                    "Rastreamento": True,
                                    "Fadiga": False,
                                    "Bocejo": False,
                                    "Postura": False,
                                    "Desatencao": False,
                                },
                            )
                            draw_face_box(frame, face, YELLOW)
                            if self.config.showcase and calibrator.active(now):
                                draw_calibration_meter(frame, calibrator.elapsed(now), self.config.calibration_seconds)
                            metrics_rows = [
                                ("Face", "detectada"),
                                ("Amostra", analysis.quality_note),
                                ("Baseline", f"{baseline.samples} amostras"),
                            ]
                            statuses = [("Rastreamento", True), ("Metricas confiaveis", False)]
                            logger.write(
                                timestamp=timestamp_iso(),
                                seconds=f"{now - start:.3f}",
                                face_detected=1,
                                tracking_quality=0,
                                quality_note=analysis.quality_note,
                                baseline_samples=baseline.samples,
                                baseline_calibrated=int(baseline.calibrated),
                                fps_current=f"{fps_stats.current:.3f}",
                                fps_mean=f"{fps_stats.mean:.3f}",
                            )
                        else:
                            calibrator.update(analysis.metrics, now)
                            smoothed_values = self.smoother.update(
                                ear=analysis.metrics.ear,
                                mar=analysis.metrics.mar,
                                yaw=analysis.metrics.yaw,
                                pitch=analysis.metrics.pitch,
                                roll=analysis.metrics.roll,
                            )
                            metrics = IndividualMetrics(**smoothed_values)
                            baseline = calibrator.baseline
                            thresholds = self._thresholds(baseline)
                            calibrating = calibrator.active(now)

                            if calibrating:
                                self.eye_closed.reset()
                                self.yawn.reset()
                                self.bad_posture.reset()
                                self.distraction.reset()
                                eye_alert = False
                                yawn_alert = False
                                posture_alert = False
                                distraction_alert = False
                            else:
                                eye_alert = self.eye_closed.update(self._eye_closed_condition(metrics, analysis, thresholds), now)
                                yawn_alert = self.yawn.update(metrics.mar > thresholds["mar"], now)
                                pitch_delta = _angle_delta(metrics.pitch, baseline.pitch)
                                yaw_delta = _angle_delta(metrics.yaw, baseline.yaw)
                                posture_alert = self.bad_posture.update(
                                    abs(pitch_delta) > self.config.posture_pitch_tolerance,
                                    now,
                                )
                                distraction_alert = self.distraction.update(
                                    abs(yaw_delta) > self.config.attention_yaw_tolerance,
                                    now,
                                )
                            fatigue_alert = eye_alert or yawn_alert

                            if fatigue_alert or posture_alert or distraction_alert:
                                self.sound.play()

                            self._write_state_transitions(
                                alert_tracker,
                                event_timeline,
                                event_logger,
                                start,
                                now,
                                {
                                    "Rastreamento": True,
                                    "Fadiga": fatigue_alert,
                                    "Bocejo": yawn_alert,
                                    "Postura": posture_alert,
                                    "Desatencao": distraction_alert,
                                },
                            )

                            draw_face_box(frame, face)
                            draw_head_pose_axis(frame, face.points, analysis.pose)
                            self._draw_alert_banner(frame, fatigue_alert, posture_alert, distraction_alert, calibrating)
                            if self.config.showcase and calibrating:
                                draw_calibration_meter(frame, calibrator.elapsed(now), self.config.calibration_seconds)

                            yaw_delta = _angle_delta(metrics.yaw, baseline.yaw)
                            pitch_delta = _angle_delta(metrics.pitch, baseline.pitch)
                            metrics_rows = [
                                ("EAR", f"{metrics.ear:.3f} (lim {thresholds['ear']:.3f})"),
                                ("Olhos E/D", f"{analysis.left_ear:.3f}/{analysis.right_ear:.3f}"),
                                ("MAR", f"{metrics.mar:.3f} (lim {thresholds['mar']:.3f})"),
                                ("Yaw delta", f"{yaw_delta:+.1f} deg"),
                                ("Pitch delta", f"{pitch_delta:+.1f} deg"),
                                ("Baseline", f"{baseline.samples} amostras"),
                            ]
                            statuses = [
                                ("Calibrando", calibrating),
                                ("Baseline OK", baseline.calibrated),
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
                                tracking_quality=1,
                                quality_note=analysis.quality_note,
                                baseline_samples=baseline.samples,
                                baseline_calibrated=int(baseline.calibrated),
                                ear=f"{metrics.ear:.5f}",
                                left_ear=f"{analysis.left_ear:.5f}",
                                right_ear=f"{analysis.right_ear:.5f}",
                                mar=f"{metrics.mar:.5f}",
                                yaw=f"{metrics.yaw:.3f}",
                                pitch=f"{metrics.pitch:.3f}",
                                roll=f"{metrics.roll:.3f}",
                                ear_threshold=f"{thresholds['ear']:.5f}",
                                mar_threshold=f"{thresholds['mar']:.5f}",
                                yaw_delta=f"{yaw_delta:.3f}",
                                pitch_delta=f"{pitch_delta:.3f}",
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
                        self._reset_tracking_state()
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
                            tracking_quality=0,
                            quality_note="landmarks insuficientes",
                            baseline_samples=calibrator.baseline.samples,
                            baseline_calibrated=int(calibrator.baseline.calibrated),
                            fps_current=f"{fps_stats.current:.3f}",
                            fps_mean=f"{fps_stats.mean:.3f}",
                        )
                    else:
                        self._reset_tracking_state()
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
                            tracking_quality=0,
                            quality_note="face ausente",
                            baseline_samples=calibrator.baseline.samples,
                            baseline_calibrated=int(calibrator.baseline.calibrated),
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

    def _reset_tracking_state(self) -> None:
        self.smoother.reset()
        self.eye_closed.reset()
        self.yawn.reset()
        self.bad_posture.reset()
        self.distraction.reset()

    @staticmethod
    def _write_state_transitions(
        alert_tracker: AlertTransitionTracker,
        event_timeline: EventTimeline,
        event_logger: CsvLogger,
        start: float,
        now: float,
        states: dict[str, bool],
    ) -> None:
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

    @staticmethod
    def _compute_analysis(points: np.ndarray, frame: np.ndarray) -> IndividualAnalysis:
        pose = estimate_head_pose(points, frame.shape)
        left_ear = eye_aspect_ratio(points, LEFT_EYE_EAR)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE_EAR)
        metrics = IndividualMetrics(
            ear=(left_ear + right_ear) / 2.0,
            mar=mouth_aspect_ratio(points),
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
        )
        quality_ok, quality_note = IndividualMode._sample_quality(metrics, pose, left_ear, right_ear)
        return IndividualAnalysis(
            metrics=metrics,
            pose=pose,
            left_ear=left_ear,
            right_ear=right_ear,
            quality_ok=quality_ok,
            quality_note=quality_note,
        )

    @staticmethod
    def _sample_quality(metrics: IndividualMetrics, pose: PoseResult, left_ear: float, right_ear: float) -> tuple[bool, str]:
        values = np.array([metrics.ear, left_ear, right_ear, metrics.mar, metrics.yaw, metrics.pitch, metrics.roll], dtype=np.float32)
        if not np.isfinite(values).all():
            return False, "metricas nao finitas"
        if not pose.success:
            return False, "pose invalida"
        if not 0.0 <= metrics.ear <= 0.45:
            return False, "EAR fora da faixa"
        if not (0.0 <= left_ear <= 0.65 and 0.0 <= right_ear <= 0.65):
            return False, "EAR unilateral instavel"
        if not 0.0 <= metrics.mar <= 0.80:
            return False, "MAR fora da faixa"
        if abs(metrics.yaw) > 110.0:
            return False, "pose extrema"
        return True, "ok"

    @staticmethod
    def _eye_closed_condition(metrics: IndividualMetrics, analysis: IndividualAnalysis, thresholds: dict[str, float]) -> bool:
        ear_threshold = thresholds["ear"]
        return metrics.ear < ear_threshold and analysis.left_ear < ear_threshold and analysis.right_ear < ear_threshold

    @staticmethod
    def _thresholds(baseline: Baseline) -> dict[str, float]:
        ear_drop = max(0.04, min(0.09, baseline.ear_mad * 3.0 + 0.025))
        ear_threshold = max(0.14, min(0.24, min(baseline.ear * 0.72, baseline.ear - ear_drop)))
        mar_margin = max(0.10, min(0.20, baseline.mar_mad * 4.0 + 0.09))
        mar_threshold = min(0.40, max(0.18, baseline.mar * 2.4, baseline.mar + mar_margin))
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
