from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from download_models import MODEL_SPECS, download


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT_DIR / "assets" / "benchmarks"
DOWNLOAD_DIR = BENCHMARK_DIR / "downloads"
WIDER_DIR = BENCHMARK_DIR / "widerface"
VIDEO_DIR = BENCHMARK_DIR / "videos"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
WIDER_FILES = {
    "WIDER_val.zip": "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip?download=true",
    "wider_face_split.zip": "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip?download=true",
}
EXPECTED_HASHES = {
    "WIDER_val.zip": "f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a",
    "wider_face_split.zip": "c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c",
    "classroom_students_ghana.webm": "29cb837026da3f2d76141c9d1ab25742046f94eb57518519049c4ca024c810f1",
    "classroom_tutoring_ghana.webm": "2db3e68da70079ca822b6a7b639da8b9573acc87bcda777a758305d8acaa9b6e",
    "classroom_learning_experience.webm": "54bbcbeae87ce5acbd6db7ec6e3ae2a360d970a97f0ad7791886dc4fab7bc375",
}
COMMONS_VIDEOS = {
    "classroom_students_ghana.webm": (
        "File:Ghanaian students learning in a dilapidated classroom.webm",
        "https://upload.wikimedia.org/wikipedia/commons/a/af/Ghanaian_students_learning_in_a_dilapidated_classroom.webm",
    ),
    "classroom_tutoring_ghana.webm": (
        "File:Students being tutored by a teacher in a dilapidated classroom.webm",
        "https://upload.wikimedia.org/wikipedia/commons/0/04/Students_being_tutored_by_a_teacher_in_a_dilapidated_classroom.webm",
    ),
    "classroom_learning_experience.webm": (
        "File:Oluyemisi's Reading Wikipedia in the Classroom Learning experience.webm",
        "https://upload.wikimedia.org/wikipedia/commons/4/44/Oluyemisi%27s_Reading_Wikipedia_in_the_Classroom_Learning_experience.webm",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa corpus grande para benchmark offline do EduTech Vision.")
    parser.add_argument("--skip-wider", action="store_true", help="Nao baixa/extrai WIDER FACE.")
    parser.add_argument("--skip-videos", action="store_true", help="Nao baixa videos reais do Wikimedia Commons.")
    parser.add_argument("--skip-research", action="store_true", help="Nao baixa modelo SCRFD de pesquisa.")
    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    model_names = ["mediapipe", "yunet"] + ([] if args.skip_research else ["scrfd"])
    for name in model_names:
        spec = MODEL_SPECS[name]
        path = Path(spec["path"])
        download(name, str(spec["url"]), path, spec.get("sha256"))
        entries.append(entry(name, path, str(spec["url"]), str(spec["license"])))

    if not args.skip_wider:
        for name, url in WIDER_FILES.items():
            path = DOWNLOAD_DIR / name
            download(name, url, path, EXPECTED_HASHES[name])
            entries.append(entry(name, path, url, "WIDER FACE academic benchmark; see dataset terms"))
            extract_zip(path, WIDER_DIR)

    if not args.skip_videos:
        for filename, (commons_title, url) in COMMONS_VIDEOS.items():
            path = VIDEO_DIR / filename
            download(filename, url, path, EXPECTED_HASHES[filename])
            license_name, page_url = commons_license(commons_title)
            entries.append(entry(filename, path, page_url, license_name, media_url=url))

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "notice": "SCRFD/InsightFace weights are for non-commercial research use; verify terms before redistribution.",
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifesto: {MANIFEST_PATH}")
    print("Corpus pronto. Os downloads permanecem locais e ignorados pelo Git.")


def extract_zip(path: Path, destination: Path) -> None:
    marker = destination / f".{path.stem}.extracted"
    if marker.exists():
        print(f"Arquivo ja extraido: {path.name}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extraindo {path.name}...")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    marker.write_text("ok\n", encoding="ascii")


def commons_license(title: str) -> tuple[str, str]:
    params = urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "extmetadata",
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
    metadata = page["imageinfo"][0].get("extmetadata", {})
    license_name = metadata.get("LicenseShortName", {}).get("value", "Wikimedia Commons license")
    return str(license_name), f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"


def entry(name: str, path: Path, source_url: str, license_name: str, **extra: str) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "path": str(path.relative_to(ROOT_DIR)),
        "source_url": source_url,
        "license": license_name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    result.update(extra)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
