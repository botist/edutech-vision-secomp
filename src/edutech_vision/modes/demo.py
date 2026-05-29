from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from edutech_vision.config import AppConfig
from edutech_vision.ui.renderer import BLUE, GREEN, RED, YELLOW, compose_dashboard, draw_wrapped_text, show_frame
from edutech_vision.utils.events import AlertTransitionTracker, EventTimeline


@dataclass(frozen=True)
class SyntheticFrame:
    seconds: float
    ear: float
    mar: float
    yaw: float
    pitch: float
    engagement: float
    faces: int
    attentive: int
    eye_alert: bool
    yawn_alert: bool
    posture_alert: bool
    distraction_alert: bool


class DemoMode:
    """Camera-free synthetic showcase for presentations."""

    def __init__(self, config: AppConfig, duration: float = 0.0) -> None:
        self.config = config
        self.duration = duration
        self.timeline = EventTimeline()
        self.transitions = AlertTransitionTracker()
        self.chart_values: deque[float] = deque(maxlen=120)

    def run(self) -> None:
        start = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                seconds = now - start
                if self.duration > 0 and seconds >= self.duration:
                    break

                data = self._synthetic_frame(seconds)
                self.chart_values.append(data.engagement)
                self._record_transitions(data)

                dashboard = self.render_dashboard(data)
                show_frame("EduTech Vision - Demo Sintetica", dashboard, fullscreen=self.config.fullscreen)
                if _quit_pressed():
                    break
                time.sleep(1 / 30)
        finally:
            cv2.destroyAllWindows()

    def _synthetic_frame(self, seconds: float) -> SyntheticFrame:
        cycle = seconds % 64.0
        eye_alert = 10.0 <= cycle < 15.0
        yawn_alert = 22.0 <= cycle < 27.0
        distraction_alert = 34.0 <= cycle < 42.0
        posture_alert = 49.0 <= cycle < 57.0

        ear = 0.29 + 0.015 * math.sin(seconds / 5.0)
        mar = 0.08 + 0.012 * math.sin(seconds / 7.0)
        yaw = 5.0 * math.sin(seconds / 4.2)
        pitch = 4.0 * math.sin(seconds / 6.0)
        if eye_alert:
            ear = 0.13 + 0.01 * math.sin(seconds * 2.0)
        if yawn_alert:
            mar = 0.30 + 0.015 * math.sin(seconds * 1.5)
        if distraction_alert:
            yaw = 29.0 + 5.0 * math.sin(seconds)
        if posture_alert:
            pitch = 20.0 + 3.0 * math.sin(seconds)

        engagement = 76.0 + 8.0 * math.sin(seconds / 8.0)
        if eye_alert or yawn_alert:
            engagement -= 11.0
        if distraction_alert:
            engagement -= 18.0
        if posture_alert:
            engagement -= 9.0
        engagement = float(np.clip(engagement, 25.0, 92.0))
        faces = 10 + int(round(2.0 * math.sin(seconds / 9.0)))
        faces = max(6, faces)
        attentive = int(round(faces * engagement / 100.0))

        return SyntheticFrame(
            seconds=seconds,
            ear=ear,
            mar=mar,
            yaw=yaw,
            pitch=pitch,
            engagement=engagement,
            faces=faces,
            attentive=attentive,
            eye_alert=eye_alert,
            yawn_alert=yawn_alert,
            posture_alert=posture_alert,
            distraction_alert=distraction_alert,
        )

    def render_still(self, seconds: float) -> np.ndarray:
        """Build a deterministic showcase frame for documentation assets."""
        self.timeline = EventTimeline()
        self.transitions = AlertTransitionTracker()
        self.chart_values.clear()
        for sample in np.linspace(0.0, seconds, 60):
            data = self._synthetic_frame(float(sample))
            self.chart_values.append(data.engagement)
            self._record_transitions(data)
        return self.render_dashboard(self._synthetic_frame(seconds))

    def render_dashboard(self, data: SyntheticFrame) -> np.ndarray:
        scene = self._draw_scene(data)
        metrics = [
            ("Fonte", "SIMULACAO"),
            ("EAR", f"{data.ear:.3f}"),
            ("MAR", f"{data.mar:.3f}"),
            ("Yaw/Pitch", f"{data.yaw:+.1f} / {data.pitch:+.1f}"),
            ("Plateia 10s", f"{data.engagement:.1f}%"),
            ("Faces", f"{data.attentive}/{data.faces} atentas"),
        ]
        statuses = [
            ("Demo guiada", True),
            ("Sem camera real", False),
            ("Fadiga/bocejo", data.eye_alert or data.yawn_alert),
            ("Desatencao", data.distraction_alert),
            ("Postura", data.posture_alert),
        ]
        bars = [
            ("EAR olho", data.ear, 0.0, 0.38, RED if data.eye_alert else GREEN),
            ("MAR boca", data.mar, 0.0, 0.45, YELLOW if data.yawn_alert else GREEN),
            ("Yaw abs", abs(data.yaw), 0.0, 40.0, RED if data.distraction_alert else BLUE),
            ("Engajamento", data.engagement, 0.0, 100.0, GREEN if data.engagement >= 60 else YELLOW),
        ]
        return compose_dashboard(
            scene,
            "Demo Sintetica",
            metrics,
            statuses,
            "Sem webcam | 30 FPS alvo",
            list(self.chart_values),
            bars=bars if self.config.showcase else None,
            events=self.timeline.formatted() if self.config.showcase else None,
            insight="Simulacao didatica: mostra o fluxo visual completo quando a camera nao estiver disponivel.",
            footer="Q sai | Demo sintetica",
        )

    def _record_transitions(self, data: SyntheticFrame) -> None:
        states = {
            "Fadiga": data.eye_alert,
            "Bocejo": data.yawn_alert,
            "Postura": data.posture_alert,
            "Desatencao": data.distraction_alert,
            "Plateia baixa": data.engagement < 55.0,
        }
        for name, active in self.transitions.update(states):
            detail = "simulado acionado" if active else "simulado normalizado"
            self.timeline.add(data.seconds, name, detail)

    def _draw_scene(self, data: SyntheticFrame) -> np.ndarray:
        width = self.config.width
        height = self.config.height
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (28, 32, 38)
        cv2.rectangle(frame, (0, 0), (width, 46), (45, 18, 18), -1)
        cv2.putText(
            frame,
            "DEMO SINTETICA - FLUXO COMPLETO EM TEMPO REAL",
            (28, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        self._draw_individual_panel(frame, data)
        self._draw_audience_panel(frame, data)
        return frame

    def _draw_individual_panel(self, frame: np.ndarray, data: SyntheticFrame) -> None:
        height, width = frame.shape[:2]
        panel_x = 42
        panel_y = 78
        panel_w = int(width * 0.42)
        panel_h = height - 120
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (42, 48, 55), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (84, 93, 105), 1)
        cv2.putText(frame, "Modo Individual sintetico", (panel_x + 22, panel_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

        center = (panel_x + panel_w // 2, panel_y + panel_h // 2 + 12)
        face_radius = min(panel_w, panel_h) // 4
        cv2.circle(frame, center, face_radius, (58, 72, 88), -1)
        cv2.circle(frame, center, face_radius, (130, 150, 170), 2)

        eye_gap = face_radius // 2
        eye_y = center[1] - face_radius // 4
        eye_h = max(2, int(data.ear * 42))
        for eye_x in (center[0] - eye_gap, center[0] + eye_gap):
            cv2.ellipse(frame, (eye_x, eye_y), (24, eye_h), 0, 0, 360, (70, 210, 245), 2)
            cv2.circle(frame, (eye_x + int(data.yaw / 3.0), eye_y), 4, (235, 235, 235), -1)

        mouth_w = face_radius
        mouth_h = max(4, int(data.mar * 120))
        cv2.ellipse(frame, (center[0], center[1] + face_radius // 3), (mouth_w // 2, mouth_h), 0, 0, 180, (80, 220, 120), 2)

        axis_len = face_radius
        yaw_offset = int(data.yaw * 1.4)
        pitch_offset = int(data.pitch * 1.1)
        cv2.arrowedLine(frame, center, (center[0] + yaw_offset, center[1] - axis_len // 2), (240, 170, 70), 2)
        cv2.arrowedLine(frame, center, (center[0], center[1] + pitch_offset), (70, 70, 240), 2)

        draw_wrapped_text(
            frame,
            "Geometria desenhada em tempo real: EAR, MAR, yaw e pitch variam por senoides e eventos programados.",
            panel_x + 22,
            panel_y + panel_h - 58,
            panel_w - 44,
            (190, 198, 208),
            0.45,
            18,
        )

    def _draw_audience_panel(self, frame: np.ndarray, data: SyntheticFrame) -> None:
        height, width = frame.shape[:2]
        panel_w = int(width * 0.42)
        panel_x = width - panel_w - 42
        panel_y = 78
        panel_h = height - 120
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (42, 48, 55), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (84, 93, 105), 1)
        cv2.putText(frame, "Modo Plateia sintetico", (panel_x + 22, panel_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

        cols = 4
        rows = math.ceil(data.faces / cols)
        spacing_x = panel_w // (cols + 1)
        spacing_y = max(48, (panel_h - 110) // max(rows, 1))
        for index in range(data.faces):
            row = index // cols
            col = index % cols
            x = panel_x + spacing_x * (col + 1)
            y = panel_y + 84 + spacing_y * row
            attentive = index < data.attentive
            color = (80, 220, 120) if attentive else (40, 210, 245)
            cv2.circle(frame, (x, y), 20, color, 2)
            cv2.circle(frame, (x - 7, y - 4), 2, color, -1)
            cv2.circle(frame, (x + 7, y - 4), 2, color, -1)
            if attentive:
                cv2.line(frame, (x - 8, y + 7), (x + 8, y + 7), color, 2)
            else:
                cv2.line(frame, (x - 8, y + 10), (x + 8, y + 4), color, 2)

        bar_x = panel_x + 28
        bar_y = panel_y + panel_h - 48
        bar_w = panel_w - 56
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (58, 64, 72), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * data.engagement / 100.0), bar_y + 14), (80, 220, 120), -1)
        cv2.putText(frame, f"Engajamento 10s: {data.engagement:.1f}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)


def _quit_pressed() -> bool:
    key = cv2.waitKey(1) & 0xFF
    return key in {ord("q"), ord("Q"), 27}
