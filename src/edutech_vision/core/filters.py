from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class MovingAverage:
    def __init__(self, maxlen: int) -> None:
        self.values: deque[float] = deque(maxlen=max(1, maxlen))

    def update(self, value: float) -> float:
        self.values.append(float(value))
        return self.value

    @property
    def value(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self) -> None:
        self.values.clear()


@dataclass
class MetricSmoother:
    window: int
    _filters: dict[str, MovingAverage] = field(default_factory=dict)

    def update(self, **metrics: float) -> dict[str, float]:
        smoothed: dict[str, float] = {}
        for name, value in metrics.items():
            smoother = self._filters.setdefault(name, MovingAverage(self.window))
            smoothed[name] = smoother.update(value)
        return smoothed

    def reset(self) -> None:
        for smoother in self._filters.values():
            smoother.reset()


class SustainedCondition:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.started_at: float | None = None

    def update(self, condition: bool, now: float) -> bool:
        if not condition:
            self.started_at = None
            return False
        if self.started_at is None:
            self.started_at = now
            return False
        return now - self.started_at >= self.seconds

    def reset(self) -> None:
        self.started_at = None
