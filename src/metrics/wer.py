from src.metrics.base_metric import (
    ArgmaxMetric,
    BeamSearchMetric,
    CustomBeamSearchMetric,
)
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(ArgmaxMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, binary_func=calc_wer, **kwargs)


class BeamSearchWERMetric(BeamSearchMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, binary_func=calc_wer, **kwargs)


class CustomBeamSearchWERMetric(CustomBeamSearchMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, binary_func=calc_wer, **kwargs)
