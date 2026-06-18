from __future__ import annotations

import pytest

from edutech_vision.core.filters import MetricSmoother
from edutech_vision.core.models import IndividualMetrics, PoseResult
from edutech_vision.modes.individual import Baseline, BaselineCalibrator, IndividualAnalysis, IndividualMode, _angle_delta, _frontal_pitch_delta


def test_metric_smoother_reset_discards_stale_values() -> None:
    smoother = MetricSmoother(window=3)
    assert smoother.update(ear=0.30)["ear"] == 0.30
    assert smoother.update(ear=0.12)["ear"] == 0.21

    smoother.reset()

    assert smoother.update(ear=0.28)["ear"] == 0.28


def test_baseline_calibrator_ignores_non_neutral_samples() -> None:
    calibrator = BaselineCalibrator(duration_seconds=5.0)
    now = calibrator.started_at + 1.0

    for _ in range(12):
        calibrator.update(IndividualMetrics(ear=0.30, mar=0.06, yaw=4.0, pitch=2.0, roll=0.0), now)
    calibrator.update(IndividualMetrics(ear=0.09, mar=0.06, yaw=4.0, pitch=2.0, roll=0.0), now)
    calibrator.update(IndividualMetrics(ear=0.30, mar=0.35, yaw=4.0, pitch=2.0, roll=0.0), now)

    baseline = calibrator.baseline

    assert baseline.calibrated is True
    assert baseline.samples == 12
    assert baseline.ear == pytest.approx(0.30)
    assert baseline.mar == pytest.approx(0.06)


def test_baseline_calibrator_accepts_wrapped_frontal_pitch() -> None:
    calibrator = BaselineCalibrator(duration_seconds=5.0)
    now = calibrator.started_at + 1.0

    for index in range(12):
        pitch = -179.0 if index % 2 == 0 else 179.0
        calibrator.update(IndividualMetrics(ear=0.29, mar=0.05, yaw=2.0, pitch=pitch, roll=0.0), now)

    baseline = calibrator.baseline

    assert baseline.calibrated is True
    assert baseline.samples == 12
    assert abs(_angle_delta(-179.0, baseline.pitch)) < 2.0
    assert abs(_angle_delta(179.0, baseline.pitch)) < 2.0


def test_baseline_calibrator_waits_for_first_valid_sample() -> None:
    calibrator = BaselineCalibrator(duration_seconds=5.0)
    late = calibrator.started_at + 30.0

    assert calibrator.active(late) is True
    assert calibrator.elapsed(late) == 0.0

    for index in range(12):
        calibrator.update(IndividualMetrics(ear=0.29, mar=0.05, yaw=1.0, pitch=-179.0, roll=0.0), late + index * 0.1)

    assert calibrator.baseline.calibrated is True
    assert calibrator.active(late + 1.0) is True
    assert calibrator.active(late + 6.0) is False


def test_individual_thresholds_are_adaptive_but_bounded() -> None:
    thresholds = IndividualMode._thresholds(Baseline(ear=0.30, mar=0.07, ear_mad=0.005, mar_mad=0.01, samples=20))

    assert 0.20 <= thresholds["ear"] <= 0.22
    assert 0.18 <= thresholds["mar"] <= 0.22


def test_sample_quality_rejects_invalid_pose() -> None:
    ok, note = IndividualMode._sample_quality(
        IndividualMetrics(ear=0.30, mar=0.07, yaw=0.0, pitch=0.0, roll=0.0),
        PoseResult(success=False),
        0.30,
        0.30,
    )

    assert ok is False
    assert "pose" in note


def test_eye_closed_condition_requires_both_eyes() -> None:
    metrics = IndividualMetrics(ear=0.19, mar=0.07, yaw=0.0, pitch=0.0, roll=0.0)
    one_eye = IndividualAnalysis(
        metrics=metrics,
        pose=PoseResult(success=True),
        left_ear=0.09,
        right_ear=0.29,
        quality_ok=True,
        quality_note="ok",
    )
    both_eyes = IndividualAnalysis(
        metrics=IndividualMetrics(ear=0.10, mar=0.07, yaw=0.0, pitch=0.0, roll=0.0),
        pose=PoseResult(success=True),
        left_ear=0.09,
        right_ear=0.11,
        quality_ok=True,
        quality_note="ok",
    )

    assert IndividualMode._eye_closed_condition(metrics, one_eye, {"ear": 0.20}) is False
    assert IndividualMode._eye_closed_condition(both_eyes.metrics, both_eyes, {"ear": 0.20}) is True


def test_angle_delta_handles_pitch_wraparound() -> None:
    assert _angle_delta(-178.0, 179.0) == pytest.approx(3.0)
    assert _angle_delta(179.0, -178.0) == pytest.approx(-3.0)
    assert abs(_frontal_pitch_delta(-179.0)) == pytest.approx(1.0)
    assert abs(_frontal_pitch_delta(179.0)) == pytest.approx(1.0)
