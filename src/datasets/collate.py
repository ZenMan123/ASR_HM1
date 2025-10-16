import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    audios = [it["audio"].squeeze(0) for it in dataset_items]
    specs = [it["spectrogram"] for it in dataset_items]
    texts = [it["text"] for it in dataset_items]
    paths = [it["audio_path"] for it in dataset_items]

    txt_ids = [
        it["text_encoded"].squeeze(0).to(dtype=torch.long) for it in dataset_items
    ]

    audio_len = torch.tensor([a.numel() for a in audios], dtype=torch.long)
    spec_len = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    txt_len = torch.tensor([t.numel() for t in txt_ids], dtype=torch.long)

    audio_batch = pad_sequence(audios, batch_first=True, padding_value=0.0)

    specs_TF = [s.squeeze(0).transpose(0, 1) for s in specs]
    specs_padded = pad_sequence(specs_TF, batch_first=True, padding_value=0.0)
    spec_batch = specs_padded.transpose(1, 2).contiguous()

    text_encoded_batch = pad_sequence(txt_ids, batch_first=True, padding_value=0)

    batch = {
        "audio": audio_batch.float(),
        "audio_length": audio_len,
        "spectrogram": spec_batch.float(),
        "spectrogram_length": spec_len,
        "text": texts,
        "text_encoded": text_encoded_batch,
        "text_encoded_length": txt_len,
        "audio_path": paths,
    }

    return batch
