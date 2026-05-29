from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edutech_vision.protocols import run_failover_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocolo 5 - Tolerancia a falhas de webcam")
    parser.add_argument("--duration", type=float, default=75.0)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--output", type=Path, default=Path("results/failover_log.csv"))
    args = parser.parse_args()
    run_failover_protocol(args.duration, args.camera, args.output)


if __name__ == "__main__":
    main()
