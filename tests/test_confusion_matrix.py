from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edutech_vision.protocols import evaluate_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocolo 4 - Matriz de confusao")
    parser.add_argument("--labels", type=Path, default=Path("assets/samples/labels_demo.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    evaluate_confusion_matrix(args.labels, args.output_dir)


if __name__ == "__main__":
    main()
