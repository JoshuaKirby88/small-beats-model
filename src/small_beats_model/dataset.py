from pathlib import Path

import torch
from torch.utils.data import Dataset

from small_beats_model.build_dataset import DATASET_DIR
from small_beats_model.loader import MapLoader
from small_beats_model.preprocessing import (
    WINDOW_BEATS,
    AudioProcessor,
    LabelProcessor,
)
from small_beats_model.vocab import EMPTY_TOKEN


class BeatsDataset(Dataset):
    def __init__(self):
        self.indices: list[tuple[Path, int]] = []
        self.loader = MapLoader()
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()

        for meta, map_id_diff in self.loader.iter_processed_meta():
            num_windows = int(meta.total_beats / WINDOW_BEATS)
            map_dir = DATASET_DIR / map_id_diff
            self.indices.extend(map(lambda i: (map_dir, i), range(num_windows)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, global_index: int):
        (map_dir, window_i) = self.indices[global_index]

        meta = self.loader.load_meta(map_dir)

        audio_tensor: torch.Tensor = torch.load(map_dir / "features.pt")
        audio_tensor_slice = self.audio_processor.slice_audio_tensor(
            audio_tensor, meta.bpm, window_i
        )

        label_tensor: torch.Tensor = torch.load(map_dir / "labels.pt")
        label_tensor_slice = self.label_processor.slice_label_tensor(
            label_tensor, window_i
        )

        start_token = torch.tensor([EMPTY_TOKEN], dtype=torch.long)
        prev_tokens = torch.cat([start_token, label_tensor_slice[:-1]])

        return (
            audio_tensor_slice,
            prev_tokens,
            label_tensor_slice,
        )
