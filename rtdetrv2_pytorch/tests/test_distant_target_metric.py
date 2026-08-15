"""The distant-target metric must count the right boxes and resist being gamed.

The second property is the reason this metric exists. COCO's `AR` applies no score threshold,
so a model that scatters weak boxes scores well on it and is useless in deployment. That is not
hypothetical: on the band training run, the epoch `AR_small` selected came LAST on the range
ladder. These tests pin the behaviour that avoids repeating it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.misc.distant_target_metric import (  # noqa: E402
    better,
    distant_target_recall,
    width_ladder,
)

NARROW = [100.0, 50.0, 106.0, 54.0]   # 6 px wide, inside the 4-12 contested range
WIDE = [500.0, 50.0, 530.0, 70.0]     # 30 px wide, outside it


@pytest.fixture
def ground_truth():
    return {1: [NARROW, WIDE]}


def test_given_boxes_outside_the_width_range_when_scored_then_they_are_not_counted(ground_truth):
    result = distant_target_recall(ground_truth, {1: [NARROW, WIDE]})

    assert result.total == 1, "only the 6 px box is in the contested range"
    assert result.recall == pytest.approx(1.0)


def test_given_the_narrow_target_is_missed_then_recall_is_zero(ground_truth):
    result = distant_target_recall(ground_truth, {1: [WIDE]})

    assert result.total == 1
    assert result.recall == pytest.approx(0.0)


def test_given_a_model_that_floods_the_frame_then_it_does_not_beat_a_clean_one(ground_truth):
    clean = distant_target_recall(ground_truth, {1: [NARROW, WIDE]})
    flood = distant_target_recall(
        ground_truth,
        {1: [NARROW] + [[float(x), 200.0, float(x) + 5, 204.0] for x in range(0, 400, 10)]},
    )

    assert flood.recall == pytest.approx(clean.recall), "same recall"
    assert flood.boxes_per_image > clean.boxes_per_image * 10, "bought with far more boxes"
    assert not better(flood, clean), "so it must not win"


def test_given_a_better_model_then_it_wins(ground_truth):
    found = distant_target_recall(ground_truth, {1: [NARROW, WIDE]})
    missed = distant_target_recall(ground_truth, {1: [WIDE]})

    assert better(found, missed)


def test_given_a_loose_overlap_then_the_target_counts_as_found(ground_truth):
    # Two pixels off a 6 px box: the tracker refines extent, so this is a detection.
    nudged = {1: [[102.0, 50.0, 108.0, 54.0]]}

    assert distant_target_recall(ground_truth, nudged).recall == pytest.approx(1.0)
    assert distant_target_recall(ground_truth, nudged, iou_threshold=0.9).recall == pytest.approx(0.0)


def test_given_a_ladder_then_each_box_lands_in_its_width_bucket(ground_truth):
    rows = width_ladder(ground_truth, {1: [NARROW, WIDE]})

    assert [(bucket, total) for bucket, total, _ in rows] == [((6, 8), 1), ((20, 40), 1)]
