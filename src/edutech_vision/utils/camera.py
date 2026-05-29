from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class ResilientVideoSource:
    def __init__(self, source: int | str, width: int = 1280, height: int = 720) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None
        self.last_reconnect_attempt = 0.0
        self.reconnect_interval = 1.0
        self.open()

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int)

    def open(self) -> bool:
        self.release()
        if self.is_camera:
            self.capture = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(int(self.source))
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            path = str(Path(str(self.source)))
            self.capture = cv2.VideoCapture(path)
        return bool(self.capture and self.capture.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.capture or not self.capture.isOpened():
            self._try_reconnect()
            return False, None

        ok, frame = self.capture.read()
        if ok and frame is not None:
            return True, frame

        if not self.is_camera:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            return bool(ok and frame is not None), frame if ok else None

        self.release()
        self._try_reconnect()
        return False, None

    def _try_reconnect(self) -> None:
        now = time.monotonic()
        if now - self.last_reconnect_attempt >= self.reconnect_interval:
            self.last_reconnect_attempt = now
            self.open()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None


def blank_frame(width: int, height: int, message: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    cv2.putText(frame, "Pressione Q para sair", (40, height // 2 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return frame
