from __future__ import annotations

from edutech_vision.launcher import visible_process_line


def test_control_center_filters_known_mediapipe_native_noise_only() -> None:
    assert not visible_process_line("W0000 face_landmarker_graph.cc:180 acceleration")
    assert not visible_process_line("E0000 portable_clearcut_uploader.cc:90 failed")
    assert visible_process_line("RuntimeError: Modelo YuNet ausente")
