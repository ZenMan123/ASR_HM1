from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                transc_path = Path(transcription_dir) / (path.stem + ".txt")
                if transc_path.exists():
                    with transc_path.open() as f:
                        entry["text"] = f.read().strip()

            info = torchaudio.info(str(path))
            num_frames = getattr(info, "num_frames", None)
            sample_rate = getattr(info, "sample_rate", None)
            length = num_frames / sample_rate
            entry["audio_len"] = length

            data.append(entry)
        super().__init__(data, *args, **kwargs)
