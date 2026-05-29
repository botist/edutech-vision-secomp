from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edutech_vision.protocols import run_lighting_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocolo 1 - Robustez luminosa")
    parser.add_argument("--mode", choices=("individual", "plateia"), default="individual")
    parser.add_argument("--stage-duration", type=float, default=20.0)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--video")
    parser.add_argument("--output", type=Path, default=Path("results/lighting_evaluation.csv"))
    args = parser.parse_args()
    run_lighting_protocol(args.mode, args.stage_duration, args.camera, args.video, args.output)


if __name__ == "__main__":
    main()
