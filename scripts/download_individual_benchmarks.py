from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2

from download_models import download, sha256


ROOT_DIR = Path(__file__).resolve().parents[1]
INDIVIDUAL_DIR = ROOT_DIR / "assets" / "benchmarks" / "individual"
VIDEO_DIR = INDIVIDUAL_DIR / "videos"
IMAGE_DIR = INDIVIDUAL_DIR / "images"
MANIFEST_PATH = INDIVIDUAL_DIR / "manifest.json"


@dataclass(frozen=True)
class CommonsAsset:
    name: str
    title: str
    kind: str
    notes: str


ASSETS = [
    CommonsAsset(
        name="nasa_jamie_brock_spherex_1080p.webm",
        title="File:NASA Interview Opportunity- Two Missions, One Rocket- One Shared Goal (SVS14783 - Jamie Brock SPHEREx).webm",
        kind="video",
        notes="Front-facing talking-head interview, 1080p, public domain.",
    ),
    CommonsAsset(
        name="nasa_cristian_parker_interview_1080p.webm",
        title="File:NASA Interview Opportunity- NASA Spacecraft Days Away From Historic Close Approach to the Sun (SVS14722 - Cristian Parker Interview).webm",
        kind="video",
        notes="Front-facing talking-head interview, 1080p, public domain.",
    ),
    CommonsAsset(
        name="lakhan_lal_interview_1080p.webm",
        title="File:Interview of a Baiga tribe named Lakhan Lal in Hindi Language by Suyash Dwivedi.webm",
        kind="video",
        notes="Interview-style video, one primary frontal face, 1080p.",
    ),
    CommonsAsset(
        name="mark_forsyth_portrait.jpg",
        title="File:Mark Forsyth portrait.jpg",
        kind="image",
        notes="Centered frontal portrait.",
    ),
    CommonsAsset(
        name="kim_hunter_front_portrait.jpg",
        title="File:Kim Hunter autographed portrait (front).jpg",
        kind="image",
        notes="Front-facing portrait photo.",
    ),
    CommonsAsset(
        name="abdelkader_mesli_identity_photo.jpg",
        title="File:Abdelkader Mesli (photo d'identit\u00e9 de sa carte de d\u00e9port\u00e9 r\u00e9sistant).jpg",
        kind="image",
        notes="Identity-style frontal photo.",
    ),
    CommonsAsset(
        name="moyomesto_passport_photo.jpg",
        title="File:Moyomesto Passport Photo.jpg",
        kind="image",
        notes="Passport-style frontal photo.",
    ),
]


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for asset in ASSETS:
        info = commons_info(asset.title)
        output_dir = VIDEO_DIR if asset.kind == "video" else IMAGE_DIR
        path = output_dir / asset.name
        download(asset.name, info["url"], path)
        metadata = video_metadata(path) if asset.kind == "video" else image_metadata(path)
        entries.append(
            {
                "name": asset.name,
                "kind": asset.kind,
                "path": str(path.relative_to(ROOT_DIR)),
                "source_url": info["description_url"],
                "media_url": info["url"],
                "license": info["license"],
                "notes": asset.notes,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                **metadata,
            }
        )

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "notice": "Individual-mode benchmark media remains local and ignored by Git.",
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifesto individual: {MANIFEST_PATH}")


def commons_info(title: str) -> dict[str, str]:
    params = urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        }
    )
    request = Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        headers={"User-Agent": "EduTechVision/1.0 (https://github.com/botist/edutech-vision-secomp)"},
    )
    with urlopen(request) as response:
        data = json.load(response)
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page or not page.get("imageinfo"):
        raise RuntimeError(f"Arquivo nao encontrado no Commons: {title}")
    imageinfo = page["imageinfo"][0]
    metadata = imageinfo.get("extmetadata", {})
    return {
        "url": str(imageinfo["url"]),
        "description_url": str(imageinfo.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"),
        "license": str(metadata.get("LicenseShortName", {}).get("value", "Wikimedia Commons license")),
    }


def image_metadata(path: Path) -> dict[str, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Nao foi possivel abrir imagem: {path}")
    height, width = image.shape[:2]
    return {"width": width, "height": height}


def video_metadata(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(frames / fps, 3) if fps > 0 else 0.0,
    }


if __name__ == "__main__":
    main()
