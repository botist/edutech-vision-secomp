from __future__ import annotations

from pathlib import Path

import cv2

from edutech_vision.config import AppConfig
from edutech_vision.modes.demo import DemoMode


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "poster" / "demo_dashboard.png"


def main() -> None:
    config = AppConfig(width=1120, height=630, showcase=True, sound_enabled=False)
    frame = DemoMode(config).render_still(37.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), frame):
        raise RuntimeError(f"Nao foi possivel gerar {OUTPUT}.")
    print(f"Captura gerada: {OUTPUT}")


if __name__ == "__main__":
    main()
