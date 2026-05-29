from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EngagementSample:
    timestamp: float
    faces: int
    attentive: int


class EngagementWindow:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.samples: deque[EngagementSample] = deque()

    def update(self, timestamp: float, faces: int, attentive: int) -> float:
        self.samples.append(EngagementSample(timestamp, faces, attentive))
        self._drop_old(timestamp)
        return self.engagement_percent

    @property
    def engagement_percent(self) -> float:
        total_faces = sum(sample.faces for sample in self.samples)
        total_attentive = sum(sample.attentive for sample in self.samples)
        if total_faces <= 0:
            return 0.0
        return 100.0 * total_attentive / total_faces

    @property
    def mean_faces(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.faces for sample in self.samples) / len(self.samples)

    @property
    def mean_attentive(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.attentive for sample in self.samples) / len(self.samples)

    def _drop_old(self, timestamp: float) -> None:
        threshold = timestamp - self.seconds
        while self.samples and self.samples[0].timestamp < threshold:
            self.samples.popleft()
