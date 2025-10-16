from abc import abstractmethod
from typing import List

import torch
from pyctcdecode import build_ctcdecoder
from torch import Tensor

from src.metrics.utils import expand_and_merge_beams, truncate_beams


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


class ArgmaxMetric(BaseMetric):
    def __init__(self, text_encoder, binary_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.binary_func = binary_func

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        results = []
        predictions = torch.argmax(log_probs.detach().cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            results.append(self.binary_func(target_text, pred_text))
        return sum(results) / len(results)


class BeamSearchMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=20, binary_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.EMPTY_TOK = text_encoder.EMPTY_TOK
        self.binary_func = binary_func

        ind2char = {i: self.text_encoder[i] for i in range(len(self.text_encoder))}
        VOCAB = [ind2char[i] for i in range(len(ind2char))]

        ctcdecoder_args = {}
        if "use_lm" in kwargs and kwargs["use_lm"]:
            from torchaudio.models.decoder import download_pretrained_files

            files = download_pretrained_files("librispeech-4-gram")
            ctcdecoder_args["kenlm_model_path"] = files.lm

            if "alpha" in kwargs:
                ctcdecoder_args["alpha"] = kwargs["alpha"]
            if "beta" in kwargs:
                ctcdecoder_args["beta"] = kwargs["beta"]

        self.decoder = build_ctcdecoder(VOCAB, **ctcdecoder_args)

    def __call__(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: List[str],
        **kwargs,
    ):
        results = []
        for i, (T, target_text) in enumerate(
            zip(log_probs_length.detach().cpu().tolist(), text)
        ):
            logits = log_probs[i, :T, :].detach().cpu().numpy()
            pred_text = self.decoder.decode(logits, beam_width=self.beam_size)

            if getattr(self.text_encoder, "SPACE_PIECE", False):
                pred_text = pred_text.replace(self.text_encoder.SPACE_PIECE, " ")

            ref = self.text_encoder.normalize_text(target_text)
            results.append(self.binary_func(ref, pred_text))
        return sum(results) / len(results)


class CustomBeamSearchMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=20, binary_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.EMPTY_TOK = text_encoder.EMPTY_TOK
        self.binary_func = binary_func

        ind2char = {i: self.text_encoder[i] for i in range(len(self.text_encoder))}
        self.VOCAB = [ind2char[i] for i in range(len(ind2char))]

    def __call__(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: List[str],
        **kwargs,
    ):
        results = []

        for i, (T, target_text) in enumerate(
            zip(log_probs_length.detach().cpu().tolist(), text)
        ):
            probs = log_probs[i, :T, :].detach().cpu().exp()
            dp = {("", self.EMPTY_TOK): 1.0}
            for t in range(T):
                cur_step_prob = probs[t]
                dp = expand_and_merge_beams(
                    dp, cur_step_prob, self.VOCAB, self.EMPTY_TOK
                )
                dp = truncate_beams(dp, self.beam_size)

            hypos = [(pref, proba) for (pref, _), proba in dp.items()]
            hypos.sort(key=lambda x: -x[1])
            pred_text = hypos[0][0] if hypos else ""

            if getattr(self.text_encoder, "SPACE_PIECE", False):
                pred_text = pred_text.replace(self.text_encoder.SPACE_PIECE, " ")

            ref = self.text_encoder.normalize_text(target_text)
            results.append(self.binary_func(ref, pred_text))

        return sum(results) / len(results)
