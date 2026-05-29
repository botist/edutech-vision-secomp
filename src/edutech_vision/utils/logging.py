from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._existing_header_mismatches():
            archived = self.path.with_name(f"{self.path.stem}.legacy-{datetime.now().strftime('%Y%m%d-%H%M%S')}{self.path.suffix}")
            self.path.replace(archived)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if needs_header:
            self.writer.writeheader()

    def _existing_header_mismatches(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                return False
        return header != self.fieldnames

    def write(self, **row: Any) -> None:
        clean = {name: row.get(name, "") for name in self.fieldnames}
        self.writer.writerow(clean)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def timestamp_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
