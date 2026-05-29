from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimelineEvent:
    seconds: float
    label: str
    detail: str

    def format(self) -> str:
        return f"{self.seconds:05.1f}s - {self.label}: {self.detail}"


@dataclass
class EventTimeline:
    maxlen: int = 8
    events: deque[TimelineEvent] = field(init=False)

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.maxlen)

    def add(self, seconds: float, label: str, detail: str) -> TimelineEvent:
        event = TimelineEvent(seconds=seconds, label=label, detail=detail)
        self.events.append(event)
        return event

    def formatted(self) -> list[str]:
        return [event.format() for event in self.events]


class AlertTransitionTracker:
    def __init__(self) -> None:
        self.previous: dict[str, bool] = {}

    def update(self, states: dict[str, bool]) -> list[tuple[str, bool]]:
        transitions: list[tuple[str, bool]] = []
        for name, active in states.items():
            previous = self.previous.get(name, False)
            if active != previous:
                transitions.append((name, active))
            self.previous[name] = active
        return transitions
