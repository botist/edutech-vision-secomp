from __future__ import annotations

import edutech_vision.core.landmarks as landmarks


class _FakeLandmarker:
    def __init__(self, **_: object) -> None:
        pass

    def close(self) -> None:
        pass


def test_enhanced_multiscale_is_reserved_for_single_face_mode(monkeypatch) -> None:
    multiscale_values: list[bool] = []

    class FakeYuNet:
        def __init__(self, *args: object, **kwargs: object) -> None:
            multiscale_values.append(bool(kwargs["multiscale"]))

    monkeypatch.setattr(landmarks, "_MediaPipeLandmarker", _FakeLandmarker)
    monkeypatch.setattr(landmarks, "YuNetFaceDetector", FakeYuNet)

    individual = landmarks.FaceLandmarkDetector(max_faces=1, detector_profile="enhanced")
    audience = landmarks.FaceLandmarkDetector(max_faces=24, detector_profile="enhanced")
    individual.close()
    audience.close()

    assert multiscale_values == [True, False]


def test_research_profile_forwards_scrfd_input_size(monkeypatch) -> None:
    input_sizes: list[int] = []

    class FakeScrfd:
        def __init__(self, *args: object, **kwargs: object) -> None:
            input_sizes.append(int(kwargs["input_size"]))

    monkeypatch.setattr(landmarks, "_MediaPipeLandmarker", _FakeLandmarker)
    monkeypatch.setattr(landmarks, "ScrfdFaceDetector", FakeScrfd)

    detector = landmarks.FaceLandmarkDetector(
        max_faces=24,
        detector_profile="research",
        scrfd_input_size=960,
    )
    detector.close()

    assert input_sizes == [960]
