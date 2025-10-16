from src.metrics.base_metric import (
    ArgmaxMetric,
    BeamSearchMetric,
    CustomBeamSearchMetric,
)
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(ArgmaxMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, binary_func=calc_cer, **kwargs)


class BeamSearchCERMetric(BeamSearchMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, binary_func=calc_cer, **kwargs)


class CustomBeamSearchCERMetric(CustomBeamSearchMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, binary_func=calc_cer, **kwargs)
