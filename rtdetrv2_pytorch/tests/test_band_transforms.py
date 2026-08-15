"""The horizon band crops must keep native scale and carry the boxes with them.

These guard a break that is silent rather than loud. torchvision renamed the Transform hooks
between 0.19 and 0.20, and a transform that implements the wrong pair is simply never called,
so the crop quietly does nothing and training runs on the full strip.
"""

import sys
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data._misc import convert_to_tv_tensor  # noqa: E402
from src.data.transforms import CenterBandCrop, RandomBandCrop  # noqa: E402

STRIP_HEIGHT, STRIP_WIDTH, BAND_HEIGHT = 640, 2560, 320
MID_BOX_TOP = 300.0


def strip_sample():
    """A stored strip with one box mid-band and one outside it, in XYXY."""
    image = PILImage.new("RGB", (STRIP_WIDTH, STRIP_HEIGHT))
    boxes = torch.tensor([[100.0, MID_BOX_TOP, 110.0, 310.0], [50.0, 10.0, 60.0, 20.0]])
    return image, convert_to_tv_tensor(
        boxes, key="boxes", box_format="XYXY", spatial_size=(STRIP_HEIGHT, STRIP_WIDTH)
    )


@pytest.mark.parametrize("crop", [CenterBandCrop(BAND_HEIGHT), RandomBandCrop(BAND_HEIGHT)])
def test_given_a_tall_strip_when_cropped_then_the_band_is_network_height_and_full_width(crop):
    image, boxes = strip_sample()

    cropped_image, cropped_boxes = crop(image, boxes)

    assert F.get_size(cropped_image) == [BAND_HEIGHT, STRIP_WIDTH]
    assert cropped_boxes.canvas_size == (BAND_HEIGHT, STRIP_WIDTH)


def test_given_a_centre_crop_when_applied_then_boxes_shift_by_the_window_offset():
    image, boxes = strip_sample()

    _, cropped_boxes = CenterBandCrop(BAND_HEIGHT)(image, boxes)

    offset = (STRIP_HEIGHT - BAND_HEIGHT) // 2
    assert float(cropped_boxes[0][1]) == pytest.approx(MID_BOX_TOP - offset)


def test_given_repeated_random_crops_then_the_window_moves():
    image, boxes = strip_sample()
    crop = RandomBandCrop(BAND_HEIGHT)

    rows = {float(crop(image, boxes)[1][0][1]) for _ in range(30)}

    assert len(rows) > 1


def test_given_no_jitter_allowed_then_the_random_crop_starts_at_the_top():
    image, boxes = strip_sample()

    _, cropped_boxes = RandomBandCrop(BAND_HEIGHT, jitter_rows=0)(image, boxes)

    assert float(cropped_boxes[0][1]) == pytest.approx(MID_BOX_TOP)


def test_given_a_strip_shorter_than_the_band_then_the_whole_strip_is_kept():
    image = PILImage.new("RGB", (STRIP_WIDTH, 200))
    boxes = convert_to_tv_tensor(
        torch.tensor([[10.0, 20.0, 20.0, 30.0]]),
        key="boxes",
        box_format="XYXY",
        spatial_size=(200, STRIP_WIDTH),
    )

    cropped_image, _ = CenterBandCrop(BAND_HEIGHT)(image, boxes)

    assert F.get_size(cropped_image) == [200, STRIP_WIDTH]
