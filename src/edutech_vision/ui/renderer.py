from __future__ import annotations

import cv2
import numpy as np

from edutech_vision.core.models import FaceLandmarks


GREEN = (80, 220, 120)
YELLOW = (40, 210, 245)
RED = (70, 70, 240)
BLUE = (240, 170, 70)
WHITE = (235, 235, 235)
MUTED = (170, 170, 170)
PANEL_BG = (34, 38, 44)
_CONFIGURED_WINDOWS: set[tuple[str, bool]] = set()


def draw_face_box(frame: np.ndarray, face: FaceLandmarks, color: tuple[int, int, int] = GREEN) -> None:
    x, y, w, h = face.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if face.points is None:
        return
    for index in (1, 33, 61, 152, 263, 291):
        point = tuple(face.points[index].astype(int))
        cv2.circle(frame, point, 2, color, -1)


def draw_anonymous_face(frame: np.ndarray, face: FaceLandmarks, attentive: bool) -> None:
    color = GREEN if attentive else YELLOW
    x, y, w, h = face.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def show_frame(window_name: str, frame: np.ndarray, *, fullscreen: bool = False) -> None:
    key = (window_name, fullscreen)
    if key not in _CONFIGURED_WINDOWS:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])
        _CONFIGURED_WINDOWS.add(key)
    cv2.imshow(window_name, frame)


def compose_dashboard(
    frame: np.ndarray,
    title: str,
    metrics: list[tuple[str, str]],
    statuses: list[tuple[str, bool]],
    fps_text: str,
    chart_values: list[float] | None = None,
    bars: list[tuple[str, float, float, float, tuple[int, int, int]]] | None = None,
    events: list[str] | None = None,
    insight: str | None = None,
    footer: str = "",
) -> np.ndarray:
    height, width = frame.shape[:2]
    compact = height < 650
    panel_width = 560 if bars or events or insight else 390
    canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    canvas[:, :width] = frame
    canvas[:, width:] = PANEL_BG

    x0 = width + 24
    y = 32 if compact else 42
    value_x = x0 + (200 if compact else 185)
    content_width = panel_width - 52
    footer_y = height - 24
    cv2.putText(canvas, title, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52 if compact else 0.72, WHITE, 2)
    y += 24 if compact else 36
    cv2.putText(canvas, fps_text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40 if compact else 0.56, MUTED, 1)
    y += 24 if compact else 36

    if insight:
        y = draw_wrapped_text(canvas, insight, x0, y, content_width, WHITE, 0.34 if compact else 0.48, 14 if compact else 20)
        y += 4 if compact else 14

    metric_step = 23 if compact else 30
    for label, value in metrics:
        cv2.putText(canvas, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 if compact else 0.55, MUTED, 1)
        cv2.putText(canvas, value, (value_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 if compact else 0.58, WHITE, 1 if compact else 2)
        y += metric_step

    if bars:
        y += 6 if compact else 8
        bar_step = 30 if compact else 42
        for label, value, minimum, maximum, color in bars:
            draw_metric_bar(canvas, x0, y, content_width, label, value, minimum, maximum, color, compact=compact)
            y += bar_step

    y += 10 if compact else 14
    status_step = 21 if compact else 31
    event_reserve = 64 if compact and events else 0
    status_bottom = max(y, footer_y - event_reserve)
    for index, (label, active) in enumerate(statuses):
        if y + status_step > status_bottom and index < len(statuses):
            remaining = len(statuses) - index
            cv2.putText(canvas, f"+{remaining} estados", (x0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 if compact else 0.48, MUTED, 1)
            y += status_step
            break
        color = RED if active else GREEN
        cv2.circle(canvas, (x0 + 10, y - 5), 5 if compact else 7, color, -1)
        cv2.putText(canvas, label, (x0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 if compact else 0.58, WHITE, 1)
        y += status_step

    if chart_values:
        sparkline_height = 64 if compact else 90
        sparkline_y = min(height - sparkline_height - 46, y + 18)
        if sparkline_y > y - 4:
            draw_sparkline(canvas, x0, sparkline_y, panel_width - 60, sparkline_height, chart_values)
            y = sparkline_y + sparkline_height + 12

    if events:
        available = footer_y - y - 8
        line_height = 20 if compact else 36
        max_events = max(0, min(len(events), (available - 24) // line_height, 3 if compact else 5))
        if max_events > 0:
            draw_event_timeline(canvas, x0, y + 8, panel_width - 60, events, max_events=max_events, line_height=line_height)

    if footer:
        cv2.putText(
            canvas,
            footer,
            (x0, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 if compact else 0.52,
            MUTED,
            1,
        )
    return canvas


def draw_metric_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    minimum: float,
    maximum: float,
    color: tuple[int, int, int],
    *,
    compact: bool = False,
) -> None:
    bar_height = 8 if compact else 12
    cv2.putText(canvas, f"{label}: {value:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36 if compact else 0.48, MUTED, 1)
    bar_y = y + (8 if compact else 10)
    cv2.rectangle(canvas, (x, bar_y), (x + width, bar_y + bar_height), (58, 64, 72), -1)
    ratio = 0.0 if maximum <= minimum else (value - minimum) / (maximum - minimum)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(canvas, (x, bar_y), (x + int(width * ratio), bar_y + bar_height), color, -1)
    cv2.rectangle(canvas, (x, bar_y), (x + width, bar_y + bar_height), (95, 102, 112), 1)


def draw_event_timeline(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    events: list[str],
    *,
    max_events: int = 5,
    line_height: int = 36,
) -> None:
    compact = line_height < 30
    cv2.putText(canvas, "Linha do tempo", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38 if compact else 0.5, MUTED, 1)
    y += 16 if compact else 24
    for event in events[-max_events:]:
        cv2.circle(canvas, (x + 7, y - 5), 4 if compact else 5, BLUE, -1)
        draw_wrapped_text(canvas, event, x + 20, y, width - 22, WHITE, 0.34 if compact else 0.45, 13 if compact else 18)
        y += line_height


def draw_wrapped_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    width: int,
    color: tuple[int, int, int],
    scale: float,
    line_height: int,
) -> int:
    words = text.split()
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        size = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if size > width and line:
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
            y += line_height
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y += line_height
    return y


def draw_calibration_meter(frame: np.ndarray, elapsed: float, total: float) -> None:
    if total <= 0:
        return
    ratio = float(np.clip(elapsed / total, 0.0, 1.0))
    width = frame.shape[1] - 80
    x, y = 40, frame.shape[0] - 46
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (35, 35, 35), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * ratio), y + 14), BLUE, -1)
    cv2.putText(frame, f"Calibracao {ratio * 100:.0f}%", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)


def draw_sparkline(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    values: list[float],
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (58, 64, 72), 1)
    if len(values) < 2:
        return
    clipped = np.clip(np.array(values, dtype=np.float32), 0, 100)
    xs = np.linspace(x + 4, x + width - 4, len(clipped)).astype(int)
    ys = (y + height - 6 - (clipped / 100.0) * (height - 12)).astype(int)
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, BLUE, 2)
    cv2.putText(canvas, "Engajamento 10s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1)
