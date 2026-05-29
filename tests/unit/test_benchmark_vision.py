from __future__ import annotations

import numpy as np

from scripts.benchmark_vision import Annotation, MetricBucket, apply_variant, match_boxes, update_bucket


def test_distance_variant_scales_ground_truth_with_image() -> None:
    image = np.ones((100, 200, 3), dtype=np.uint8)
    annotation = Annotation((20.0, 20.0, 40.0, 30.0), False, False, False, False)

    _, transformed = apply_variant(image, [annotation], "distance")

    assert transformed[0].bbox == (56.0, 33.0, 22.0, 16.5)


def test_no_face_variant_has_no_expected_faces() -> None:
    image = np.ones((80, 80, 3), dtype=np.uint8)
    annotation = Annotation((10.0, 10.0, 30.0, 30.0), False, False, False, False)

    modified, transformed = apply_variant(image, [annotation], "no_face")

    assert transformed == []
    assert np.all(modified[10:40, 10:40] == 24)


def test_box_matching_reports_unmatched_detections() -> None:
    truths = [Annotation((10.0, 10.0, 40.0, 40.0), False, False, False, False)]
    matches, missing, extras = match_boxes(truths, [(11.0, 11.0, 39.0, 39.0), (100.0, 100.0, 20.0, 20.0)])

    assert len(matches) == 1
    assert missing == []
    assert extras == [1]


def test_metric_bucket_tracks_landmark_valid_detections() -> None:
    bucket = MetricBucket()

    update_bucket(
        bucket,
        truths=2,
        predictions=2,
        matches=[(0, 0.8)],
        landmark_valid_predictions=1,
        landmark_matches=[(0, 0.8)],
    )

    assert bucket.true_positives == 1
    assert bucket.landmark_valid_predictions == 1
    assert bucket.landmark_true_positives == 1
