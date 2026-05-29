from __future__ import annotations

import time


class AlertSound:
    def __init__(self, enabled: bool = True, cooldown_seconds: float = 2.0) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_played = 0.0

    def play(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_played < self.cooldown_seconds:
            return
        self.last_played = now
        try:
            import winsound

            winsound.Beep(1200, 140)
        except Exception:
            pass
