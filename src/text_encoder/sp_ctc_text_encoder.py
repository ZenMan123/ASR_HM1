from typing import List

import sentencepiece as spm
import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import ROOT_PATH


class SPCTCTextEncoder(CTCTextEncoder):
    BLANK_ID = 0
    EMPTY_TOK = ""
    SPACE_PIECE = "▁"

    def __init__(self, sp_model_name, dataset, sp_train_kwargs=None):
        super().__init__()

        model_dir = ROOT_PATH / "data" / "tokenizers"
        model_dir.mkdir(exist_ok=True, parents=True)
        model_file = model_dir / sp_model_name
        model_file_prefix = model_file.with_suffix("")

        if not model_file.exists():

            def sentence_iter():
                for item in dataset._index:
                    s = item["text"]
                    if s:
                        yield s

            args = {
                "sentence_iterator": sentence_iter(),
                "model_prefix": str(model_file_prefix),
                "vocab_size": 4000,
                "character_coverage": 1.0,
                "model_type": "unigram",
                "hard_vocab_limit": False,
                "byte_fallback": True,
                "unk_id": 0,
                "bos_id": -1,
                "eos_id": -1,
                "pad_id": -1,
            }

            if sp_train_kwargs:
                args.update(sp_train_kwargs)

            spm.SentencePieceTrainer.train(**args)

        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.offset = 1

        self._sp_vocab_size = self.sp.get_piece_size()
        self.id2piece = {self.BLANK_ID: self.EMPTY_TOK}
        for sp_id in range(self._sp_vocab_size):
            self.id2piece[sp_id + self.offset] = self.sp.id_to_piece(sp_id)

        self.piece2id = {p: i for i, p in self.id2piece.items()}

    def __len__(self) -> int:
        return 1 + self._sp_vocab_size

    def __getitem__(self, item: int) -> str:
        assert isinstance(item, int)
        return self.id2piece[item]

    def encode(self, text: str) -> torch.Tensor:
        text = self.normalize_text(text)
        sp_ids: List[int] = self.sp.encode(text, out_type=int)
        ids = [i + self.offset for i in sp_ids]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode(self, inds) -> str:
        if isinstance(inds, list):
            seq = inds
        else:
            seq = inds.tolist()
        pieces = []
        for idx in seq:
            if idx == self.BLANK_ID:
                continue
            pieces.append(self.id2piece[idx])
        raw = "".join(pieces)
        text = raw.replace(self.SPACE_PIECE, " ").strip()
        return text

    def ctc_decode(self, inds) -> str:
        seq = inds.tolist()

        result_ids = []
        prev = None

        for idx in seq:
            if idx == self.BLANK_ID:
                prev = idx
                continue
            if prev is not None and idx == prev:
                continue
            result_ids.append(idx)
            prev = idx

        return self.decode(result_ids)
