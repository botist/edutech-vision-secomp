from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FpsStats:
    current: float
    mean: float
    std: float
    minimum: float
    maximum: float


class SlidingFPS:
    def __init__(self, seconds: float = 60.0) -> None:
        self.seconds = seconds
        self.timestamps: deque[float] = deque()
        self.instant_values: deque[float] = deque(maxlen=2000)
        self.previous_timestamp: float | None = None

    def update(self, timestamp: float) -> FpsStats:
        self.timestamps.append(timestamp)
        while self.timestamps and self.timestamps[0] < timestamp - self.seconds:
            self.timestamps.popleft()

        current = 0.0
        if self.previous_timestamp is not None:
            delta = timestamp - self.previous_timestamp
            if delta > 1e-6:
                current = 1.0 / delta
                self.instant_values.append(current)
        self.previous_timestamp = timestamp
        return self.stats(current)

    def stats(self, current: float = 0.0) -> FpsStats:
        values = np.array(self.instant_values, dtype=np.float32)
        if values.size == 0:
            return FpsStats(current=current, mean=0.0, std=0.0, minimum=0.0, maximum=0.0)
        return FpsStats(
            current=current,
            mean=float(values.mean()),
            std=float(values.std()),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )
