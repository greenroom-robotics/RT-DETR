"""Crops for the horizon band detector.

The band model reads a wide, short strip of the frame at native pixel scale, so it never
resizes. Everything else in this directory changes an image's scale; these two deliberately
do not, because native resolution is the reason the band model exists.

The dataset stores a strip taller than the network input (640 rows against 320). Training
takes a random 320-row window out of it every epoch, so the model never learns that the
horizon sits at one row. Validation takes the middle window, so a score does not move between
epochs.
"""

import PIL.Image
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from .._misc import BoundingBoxes, Image, Mask, Video
from ...core import register

_CROPPED_TYPES = (PIL.Image.Image, Image, Video, Mask, BoundingBoxes)


def _window(strip_height: int, height: int, offset: int) -> tuple[int, int]:
    """The (top, height) of a crop window, clamped inside a strip.

    A strip shorter than the requested band yields the whole strip rather than an error, so a
    mixed export still trains. The caller decides whether that is acceptable.
    """
    height = min(height, strip_height)
    top = max(0, min(offset, strip_height - height))
    return top, height


@register()
class RandomBandCrop(T.Transform):
    """A random network-height window out of the taller stored strip.

    ``jitter_rows`` caps how far the window travels from the top of the strip. It defaults to
    the full travel the strip allows, which is what stops the model learning a fixed horizon
    row. ``jitter_cols`` is normally 0: the band spans the full frame width, and cropping
    horizontally would throw away field of view the model is meant to cover.
    """

    _transformed_types = _CROPPED_TYPES

    def __init__(self, height: int, jitter_rows: int | None = None, jitter_cols: int = 0) -> None:
        super().__init__()
        self.height = height
        self.jitter_rows = jitter_rows
        self.jitter_cols = jitter_cols

    def make_params(self, flat_inputs):
        strip_height, strip_width = F.get_size(flat_inputs[0])

        travel = strip_height - self.height
        if self.jitter_rows is not None:
            travel = min(travel, self.jitter_rows)
        offset = int(torch.randint(0, travel + 1, (1,))) if travel > 0 else 0
        top, height = _window(strip_height, self.height, offset)

        width = strip_width
        left = 0
        if self.jitter_cols > 0:
            width = max(1, strip_width - self.jitter_cols)
            left = int(torch.randint(0, strip_width - width + 1, (1,)))

        return {"top": top, "left": left, "height": height, "width": width}

    def transform(self, inpt, params):
        return F.crop(inpt, **params)


@register()
class CenterBandCrop(T.Transform):
    """The middle network-height window of the stored strip.

    The deterministic counterpart of :class:`RandomBandCrop`, for validation and for any
    evaluation that has to be comparable between epochs and between runs.
    """

    _transformed_types = _CROPPED_TYPES

    def __init__(self, height: int) -> None:
        super().__init__()
        self.height = height

    def make_params(self, flat_inputs):
        strip_height, strip_width = F.get_size(flat_inputs[0])
        top, height = _window(strip_height, self.height, (strip_height - self.height) // 2)
        return {"top": top, "left": 0, "height": height, "width": strip_width}

    def transform(self, inpt, params):
        return F.crop(inpt, **params)
