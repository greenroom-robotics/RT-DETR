"""Recall of distant targets, bucketed by apparent width.

**This has nothing to do with the horizon.** It never looks at where the horizon is, uses no
INS or geopose, and works the same on imagery with no horizon in it. The horizon band detector
happens to be the model that needed it first; the metric applies to any detector.

**How much of this is just COCO reconfigured?** Most of it, and the honest accounting is:

============================  ==================================================
difference                    available in pycocotools?
============================  ==================================================
score threshold at the        **No.** ``AP`` builds a PR curve over all
deployment operating point    detections and ``AR`` takes the top 100
                              unthresholded. Neither scores where the pipeline
                              actually runs. This is the difference that matters.
single loose IoU of 0.3       Yes, ``params.iouThrs``
bucket by width, not area     Mostly, ``params.areaRng`` gets close
============================  ==================================================

Measured on the band validation set, width and sqrt(area) correlate at Spearman 0.93 and the
4-12 selection band overlaps 84.7 % between them, because the median box is nearly square
(aspect 0.89). **The width axis is a refinement, not the substance.** The substance is that
recall is measured at a usable confidence.

**Why the COCO metrics do not serve this case.** COCO buckets ground truth by *area*, with
"small" meaning under 32x32 = 1024 px². On maritime horizon data that bucket holds 89 % of all
boxes, so `AP_small` is a near-duplicate of `AP_all` and carries no information about range. It
gets worse: `AP` averages over IoU 0.50 to 0.95, and a 5x2 px box cannot reach IoU 0.9 whatever
the model does, so most of that average measures localisation the tracker will redo anyway.
And `AR` applies no score threshold, so it rewards a model that scatters weak boxes, which is
the one behaviour deployment cannot use.

**What this measures instead.** Recall of ground truth in a band of apparent *width*, at a
usable confidence, with a loose IoU. Width is the right axis because it falls with range:

    width_px = hull_length_m / (range_m * angular_resolution_rad_per_px)

so a narrower box is a more distant target of the same class. Area does not have that property,
because it mixes width and height.

**Where the bounds come from.** Measured on the Leidos EO horizon data, and they are not
arbitrary:

- Below ~4 px the target is under the sensor's floor. No model recovers it, so including that
  region adds variance and no signal.
- Above ~12 px the deployed detector already sits near 95 %. Saturated, so it dilutes.

Between them is the contested zone, which is where a change to the pipeline can still move the
number. For an 8 m hull that is roughly 1.0 to 3.1 NM; for a 25 m hull, 3.2 to 9.6 NM.

**Read it with the companion count.** Recall alone can be bought by emitting more boxes. This
returns boxes-per-image alongside, and a model that gains recall while inflating that has not
necessarily improved. On data whose labels are incomplete, and horizon labels usually are, an
unmatched box is not reliably a false positive, so the count is a guard rail rather than a
precision measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

# Apparent width in native source pixels. See the module docstring for where these come from.
CONTESTED_WIDTH_PX = (4.0, 12.0)

# The diagnostic ladder. One scalar selects a checkpoint; the ladder says WHERE a model gained
# or lost it, which a single number cannot. Edges are in apparent width, so each row is a
# range band for a given hull.
WIDTH_LADDER = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 20), (20, 40), (40, 1e9)]

# Loose on purpose. A 5x2 px box that overlaps at 0.3 has been found; the tracker refines the
# extent. Demanding 0.5 measures box regression, which is not what this metric is for.
DEFAULT_IOU = 0.3

# Detections below this are unusable downstream, so they must not earn credit here.
DEFAULT_SCORE = 0.25


@dataclass(frozen=True)
class DistantTargetRecall:
    """Recall over the contested width range, plus what it cost to get there."""

    matched: int
    total: int
    boxes_per_image: float
    width_range: tuple[float, float]
    iou_threshold: float
    score_threshold: float

    @property
    def recall(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def __str__(self) -> str:
        low, high = self.width_range
        return (
            f"horizon recall {100 * self.recall:.1f}% ({self.matched}/{self.total}) "
            f"for {low:.0f}-{high:.0f} px wide, IoU>={self.iou_threshold}, "
            f"score>={self.score_threshold}, {self.boxes_per_image:.2f} boxes/image"
        )


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two xyxy boxes."""
    overlap_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_w * overlap_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def distant_target_recall(
    ground_truth: dict[int, list[list[float]]],
    predictions: dict[int, list[list[float]]],
    width_range: tuple[float, float] = CONTESTED_WIDTH_PX,
    iou_threshold: float = DEFAULT_IOU,
    score_threshold: float = DEFAULT_SCORE,
) -> DistantTargetRecall:
    """Recall over ground truth whose width falls in *width_range*.

    Both mappings are image id to a list of xyxy boxes in the same coordinate frame. The
    predictions are expected to be thresholded already; ``score_threshold`` is carried through
    for the record so a reported number always states the operating point it belongs to.
    """
    low, high = width_range
    matched = total = 0
    for image_id, boxes in ground_truth.items():
        found = predictions.get(image_id, [])
        for box in boxes:
            if not low <= box[2] - box[0] < high:
                continue
            total += 1
            if any(iou(box, candidate) >= iou_threshold for candidate in found):
                matched += 1

    images = max(1, len(ground_truth))
    return DistantTargetRecall(
        matched=matched,
        total=total,
        boxes_per_image=sum(len(v) for v in predictions.values()) / images,
        width_range=width_range,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


def better(candidate: DistantTargetRecall, incumbent: DistantTargetRecall, box_tolerance: float = 1.15) -> bool:
    """Is *candidate* the better model to keep?

    Higher recall wins, but only if it did not buy that recall by flooding the frame. A
    candidate emitting more than ``box_tolerance`` times the incumbent's boxes has to clear it
    on recall by more than the proportional increase, which stops a scatter-everything model
    from winning on recall alone.
    """
    if incumbent.total == 0:
        return candidate.total > 0
    if candidate.boxes_per_image > incumbent.boxes_per_image * box_tolerance:
        inflation = candidate.boxes_per_image / max(1e-9, incumbent.boxes_per_image)
        return candidate.recall > incumbent.recall * inflation
    return candidate.recall > incumbent.recall


class DistantTargetAccumulator:
    """Collects predictions across an evaluation pass and scores them at the end.

    Lives in RT-DETR rather than in visionai because the training loop needs it every epoch,
    and the trainer cannot import the CLI. The CLI imports it from here instead.
    """

    def __init__(self, coco_gt, width_range=CONTESTED_WIDTH_PX, iou_threshold=DEFAULT_IOU,
                 score_threshold=DEFAULT_SCORE):
        self.coco_gt = coco_gt
        self.width_range = tuple(width_range)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.predictions: dict[int, list[list[float]]] = {}

    def update(self, results: dict) -> None:
        """*results* maps image id to the postprocessor's output, in original image pixels."""
        for image_id, output in results.items():
            boxes, scores = output["boxes"], output["scores"]
            keep = scores >= self.score_threshold
            self.predictions[int(image_id)] = boxes[keep].tolist()

    def summarize(self) -> DistantTargetRecall:
        ground_truth = {}
        for image_id in self.predictions:
            annotations = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=image_id))
            ground_truth[image_id] = [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in annotations
                if not a.get("iscrowd", 0)
            ]
        self._ground_truth = ground_truth
        return distant_target_recall(
            ground_truth,
            self.predictions,
            width_range=self.width_range,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )

    def ladder(self) -> str:
        """The per-bucket breakdown, for the log. Call after summarize()."""
        rows = width_ladder(self._ground_truth, self.predictions, self.iou_threshold)
        parts = [
            f"{int(low)}-{int(high) if high < 1e9 else '+'}px {100 * recall:.0f}%(n{total})"
            for (low, high), total, recall in rows
        ]
        return "  width ladder: " + "  ".join(parts)


def width_ladder(
    ground_truth: dict[int, list[list[float]]],
    predictions: dict[int, list[list[float]]],
    iou_threshold: float = DEFAULT_IOU,
) -> list[tuple[tuple[float, float], int, float]]:
    """Recall per apparent-width bucket: (bucket, n, recall).

    Reported for every model, used for selection by none. A model can hold its aggregate while
    trading distant targets for near ones, and only the ladder shows it.
    """
    rows = []
    for low, high in WIDTH_LADDER:
        matched = total = 0
        for image_id, boxes in ground_truth.items():
            found = predictions.get(image_id, [])
            for box in boxes:
                if not low <= box[2] - box[0] < high:
                    continue
                total += 1
                if any(iou(box, candidate) >= iou_threshold for candidate in found):
                    matched += 1
        if total:
            rows.append(((low, high), total, matched / total))
    return rows
