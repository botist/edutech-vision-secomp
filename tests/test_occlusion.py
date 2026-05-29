from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edutech_vision.protocols import run_occlusion_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocolo 2 - Resiliencia a oclusao")
    parser.add_argument("--mode", choices=("individual", "plateia"), default="individual")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--video")
    parser.add_argument("--output", type=Path, default=Path("results/occlusion_recovery.csv"))
    args = parser.parse_args()
    run_occlusion_protocol(args.mode, args.camera, args.video, args.output)


if __name__ == "__main__":
    main()
