"""Early stopping on validation mAP.
"""


class EarlyStopping(object):
    """Tracks the best val mAP and the patience counter, saved in every checkpoint.
    """

    STATE_KEYS = ('best_map', 'best_epoch', 'ref_map', 'wait', 'stopped_epoch')

    def __init__(self, enabled: bool = True, patience: int = 10, min_delta: float = 0.001) -> None:
        self.enabled = enabled
        self.patience = patience
        self.min_delta = min_delta

        self.best_map = float('-inf')
        self.best_epoch = -1
        self.ref_map = float('-inf')
        self.wait = 0
        self.stopped_epoch = None

    def step(self, value: float, epoch: int) -> bool:
        """Record one epoch's val mAP. Returns True when it is the new best."""
        is_best = value > self.best_map
        if is_best:
            self.best_map = value
            self.best_epoch = epoch

        if value - self.ref_map >= self.min_delta:
            self.ref_map = value
            self.wait = 0
        else:
            self.wait += 1

        return is_best

    @property
    def should_stop(self) -> bool:
        return self.enabled and self.wait >= self.patience

    def state_dict(self):
        return {k: getattr(self, k) for k in self.STATE_KEYS}

    def load_state_dict(self, state):
        for k in self.STATE_KEYS:
            if k in state:
                setattr(self, k, state[k])

    def __repr__(self) -> str:
        return (f'EarlyStopping(enabled={self.enabled}, patience={self.patience}, '
                f'min_delta={self.min_delta}, best_map={self.best_map}, '
                f'best_epoch={self.best_epoch}, wait={self.wait})')
