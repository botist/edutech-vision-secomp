from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "assets" / "models"
MODEL_SPECS = {
    "mediapipe": {
        "path": MODEL_DIR / "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "license": "MediaPipe model terms",
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    },
    "yunet": {
        "path": MODEL_DIR / "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "license": "MIT (OpenCV Zoo)",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "scrfd": {
        "path": MODEL_DIR / "scrfd_2.5g_bnkps.onnx",
        "url": "https://huggingface.co/laichaoyi/MixupModels/resolve/main/scrfd_2.5g_bnkps.onnx?download=true",
        "license": "InsightFace model weights: non-commercial research only",
        "sha256": "bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa modelos locais do EduTech Vision.")
    parser.add_argument("--research", action="store_true", help="Inclui SCRFD, modelo de pesquisa non-commercial.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    names = ["mediapipe", "yunet"] + (["scrfd"] if args.research else [])
    for name in names:
        spec = MODEL_SPECS[name]
        download(name, str(spec["url"]), Path(spec["path"]), spec.get("sha256"))
        print(f"Licenca {name}: {spec['license']}")

    if not args.research:
        scrfd = MODEL_SPECS["scrfd"]
        scrfd_path = Path(scrfd["path"])
        if scrfd_path.exists() and sha256(scrfd_path) == scrfd["sha256"]:
            print(f"SCRFD opcional ja disponivel: {scrfd_path}")
        else:
            print("SCRFD opcional nao instalado. Para o perfil research: python scripts/download_models.py --research")


def download(name: str, url: str, destination: Path, expected_hash: str | None = None) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        if expected_hash and sha256(destination) != expected_hash:
            destination.unlink()
        else:
            print(f"Arquivo {name} ja existe: {destination}")
            return
    print(f"Baixando {name}: {url}")
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "EduTechVision/1.0 (https://github.com/botist/edutech-vision-secomp)"})
    with urlopen(request) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if expected_hash and sha256(partial) != expected_hash:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Hash invalido para {name}. Download removido.")
    partial.replace(destination)
    print(f"Arquivo salvo em: {destination}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
