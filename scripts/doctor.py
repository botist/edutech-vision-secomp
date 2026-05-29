from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
YUNET_MODEL_PATH = ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
SCRFD_MODEL_PATH = ROOT_DIR / "assets" / "models" / "scrfd_2.5g_bnkps.onnx"
REQUIRED_PACKAGES = {
    "mediapipe": "0.10.35",
    "opencv-contrib-python": "4.11.0.86",
    "numpy": "1.26.4",
    "matplotlib": "3.10.9",
    "onnxruntime": "1.26.0",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostico rapido do ambiente EduTech Vision.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam para teste opcional.")
    parser.add_argument("--camera-check", action="store_true", help="Tenta abrir a webcam por alguns frames.")
    args = parser.parse_args()

    ok = True
    print("EduTech Vision - diagnostico")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    if not (3, 10) <= sys.version_info[:2] < (3, 12):
        ok = False
        print("[ERRO] Use Python >=3.10 e <3.12 para compatibilidade com MediaPipe.")
    else:
        print("[OK] Versao de Python compativel.")

    for package, expected in REQUIRED_PACKAGES.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            ok = False
            print(f"[ERRO] Pacote ausente: {package}")
            continue
        marker = "OK" if installed == expected else "AVISO"
        if installed != expected:
            ok = False
        print(f"[{marker}] {package}: {installed} (esperado {expected})")

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"[OK] Modelo MediaPipe encontrado: {MODEL_PATH}")
    else:
        ok = False
        print(f"[ERRO] Modelo ausente: {MODEL_PATH}")
        print(f"      Rode: {setup_command()}")

    if YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size > 0:
        print(f"[OK] Modelo YuNet encontrado: {YUNET_MODEL_PATH}")
    else:
        ok = False
        print(f"[ERRO] Modelo YuNet ausente: {YUNET_MODEL_PATH}")
        print(f"      Rode: {setup_command()}")

    if SCRFD_MODEL_PATH.exists() and SCRFD_MODEL_PATH.stat().st_size > 0:
        print(f"[INFO] Modelo SCRFD opcional encontrado: {SCRFD_MODEL_PATH}")
    else:
        print("[INFO] Modelo SCRFD opcional nao instalado; use download_models.py --research para comparacao.")

    if args.camera_check:
        ok = check_camera(args.camera) and ok

    raise SystemExit(0 if ok else 1)


def check_camera(index: int) -> bool:
    import cv2

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(f"[ERRO] Webcam {index} nao abriu.")
        return False

    good_frames = 0
    for _ in range(10):
        read_ok, frame = capture.read()
        if read_ok and frame is not None:
            good_frames += 1
    capture.release()

    if good_frames == 0:
        print(f"[ERRO] Webcam {index} abriu, mas nao retornou frames.")
        return False
    print(f"[OK] Webcam {index}: {good_frames}/10 frames lidos.")
    return True


def setup_command() -> str:
    return ".\\run.bat" if platform.system() == "Windows" else "./run.sh"


if __name__ == "__main__":
    main()
