from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from download_models import download, sha256


ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "assets" / "benchmarks" / "hq_videos"
MANIFEST_PATH = VIDEO_DIR / "manifest.json"


@dataclass(frozen=True)
class VideoSpec:
    name: str
    source_url: str
    media_url: str
    notes: str


VIDEOS = [
    VideoSpec(
        name="voa_turkish_classroom_1080p.mp4",
        source_url="https://www.voanews.com/a/learning-turkish-puts-ukrainians-russians-in-same-classroom/6974721.html",
        media_url="https://voa-video-ns.akamaized.net/pangeavideo/2023/02/0/01/01000000-0aff-0242-e47f-08db151bb860_1080p.mp4?download=1",
        notes="Classroom scene, several seated students, 1080p source.",
    ),
    VideoSpec(
        name="voa_mosul_students_1080p.mp4",
        source_url="https://www.voanews.com/a/learning-to-forget-hate-mosul-students-return-to-school-after-islamic-state/3700519.html",
        media_url="https://voa-video-ns.akamaized.net/pangeavideo/2017/01/a/aa/aa80abc3-43de-48e7-9064-044227f90188_fullhd.mp4?download=1",
        notes="Classroom and school scenes, 1080p source.",
    ),
    VideoSpec(
        name="voa_student_protest_1080p.mp4",
        source_url="https://www.voanews.com/a/anti-trump-protests-continue-focus-on-inauguration-day/3596835.html",
        media_url="https://voa-video-ns.akamaized.net/pangeavideo/2016/11/8/87/87171605-5cc7-4334-9169-d8187150baa1_fullhd.mp4?download=1",
        notes="Student crowd scene, many medium/small faces, 1080p source.",
    ),
]


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for spec in VIDEOS:
        path = VIDEO_DIR / spec.name
        download(spec.name, spec.media_url, path)
        metadata = video_metadata(path)
        entries.append(
            {
                "name": spec.name,
                "path": str(path.relative_to(ROOT_DIR)),
                "source_url": spec.source_url,
                "media_url": spec.media_url,
                "license": "Local test video; source page retained for reproducibility.",
                "notes": spec.notes,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                **metadata,
            }
        )

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "notice": "Videos are local benchmark inputs and remain ignored by Git.",
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifesto HQ: {MANIFEST_PATH}")


def video_metadata(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(duration, 3),
    }


if __name__ == "__main__":
    main()
