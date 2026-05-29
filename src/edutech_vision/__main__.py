from __future__ import annotations

import argparse
from pathlib import Path

from edutech_vision.config import AppConfig, ROOT_DIR, default_face_confidence, parse_source
from edutech_vision.modes.audience import AudienceMode
from edutech_vision.modes.demo import DemoMode
from edutech_vision.modes.individual import IndividualMode


def main() -> None:
    parser = argparse.ArgumentParser(description="EduTech Vision - Trilha 3 SECOMP/PDI")
    parser.add_argument("--mode", choices=("individual", "plateia", "demo"), help="Modo de operacao.")
    parser.add_argument("--camera", default="0", help="Indice da webcam. Padrao: 0.")
    parser.add_argument("--video", help="Arquivo de video opcional para reproducao/teste.")
    parser.add_argument("--width", type=int, default=960, help="Largura solicitada para webcam.")
    parser.add_argument("--height", type=int, default=540, help="Altura solicitada para webcam.")
    parser.add_argument("--calibration-seconds", type=float, default=5.0, help="Duracao da calibracao inicial.")
    parser.add_argument("--audience-window-seconds", type=float, default=10.0, help="Janela temporal do modo plateia.")
    parser.add_argument("--max-faces", type=int, default=24, help="Maximo de faces no modo plateia.")
    parser.add_argument("--audience-yaw-tolerance", type=float, default=30.0, help="Tolerancia de yaw no modo plateia.")
    parser.add_argument("--audience-pitch-tolerance", type=float, default=20.0, help="Tolerancia de pitch no modo plateia.")
    parser.add_argument("--attention-yaw-tolerance", type=float, default=22.0, help="Tolerancia de yaw no modo individual.")
    parser.add_argument("--posture-pitch-tolerance", type=float, default=14.0, help="Tolerancia de pitch postural no modo individual.")
    parser.add_argument("--model", help="Caminho para face_landmarker.task.")
    parser.add_argument(
        "--detector",
        choices=("mediapipe", "enhanced", "research"),
        default="enhanced",
        help="Detector: mediapipe (baseline), enhanced (YuNet + landmarks) ou research (SCRFD + landmarks).",
    )
    parser.add_argument("--face-confidence", type=float, help="Confianca minima de deteccao facial.")
    parser.add_argument("--yunet-model", help="Caminho para face_detection_yunet_2023mar.onnx.")
    parser.add_argument("--scrfd-model", help="Caminho para scrfd_2.5g_bnkps.onnx.")
    parser.add_argument("--scrfd-input-size", type=int, default=640, help="Entrada quadrada do SCRFD. Multiplo de 32.")
    parser.add_argument("--showcase", action="store_true", help="Ativa painel de apresentacao com explicabilidade e linha do tempo.")
    parser.add_argument("--demo-duration", type=float, default=0.0, help="Duracao do modo demo sintetico. 0 roda ate Q/Esc.")
    parser.add_argument("--fullscreen", action="store_true", help="Abre a janela OpenCV em tela cheia para apresentacao.")
    parser.add_argument("--no-mirror", action="store_true", help="Nao espelha imagem da webcam.")
    parser.add_argument("--no-sound", action="store_true", help="Desativa alerta sonoro no modo individual.")
    args = parser.parse_args()

    mode = args.mode or choose_mode()
    config = AppConfig(
        source=parse_source(args.camera, args.video),
        width=args.width,
        height=args.height,
        face_landmarker_model=Path(args.model) if args.model else ROOT_DIR / "assets" / "models" / "face_landmarker.task",
        yunet_model=Path(args.yunet_model) if args.yunet_model else ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx",
        scrfd_model=Path(args.scrfd_model) if args.scrfd_model else ROOT_DIR / "assets" / "models" / "scrfd_2.5g_bnkps.onnx",
        scrfd_input_size=args.scrfd_input_size,
        detector_profile=args.detector,
        face_confidence=args.face_confidence if args.face_confidence is not None else default_face_confidence(mode),
        mirror_camera=not args.no_mirror,
        calibration_seconds=args.calibration_seconds,
        audience_window_seconds=args.audience_window_seconds,
        max_audience_faces=args.max_faces,
        audience_yaw_tolerance=args.audience_yaw_tolerance,
        audience_pitch_tolerance=args.audience_pitch_tolerance,
        attention_yaw_tolerance=args.attention_yaw_tolerance,
        posture_pitch_tolerance=args.posture_pitch_tolerance,
        sound_enabled=not args.no_sound,
        showcase=args.showcase,
        fullscreen=args.fullscreen,
    )

    if mode == "individual":
        IndividualMode(config).run()
    elif mode == "plateia":
        AudienceMode(config).run()
    elif mode == "demo":
        DemoMode(config, duration=args.demo_duration).run()
    else:
        raise SystemExit(f"Modo invalido: {mode}")


def choose_mode() -> str:
    print("EduTech Vision - selecione o modo:")
    print("1 - Individual: fadiga, postura e desatencao")
    print("2 - Plateia: engajamento agregado em janela temporal")
    print("3 - Demo sintetica: fallback visual sem camera")
    while True:
        choice = input("Modo [1/2/3]: ").strip()
        if choice == "1":
            return "individual"
        if choice == "2":
            return "plateia"
        if choice == "3":
            return "demo"
        print("Opcao invalida.")


if __name__ == "__main__":
    main()
