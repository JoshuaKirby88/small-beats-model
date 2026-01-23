from pathlib import Path

import torch
from torch.utils.data import Dataset

from small_beats_model.build_dataset import DATASET_DIR
from small_beats_model.loader import MapLoader
from small_beats_model.preprocessing import (
    FPS,
    STEPS_PER_BEAT,
    TARGET_BPM,
    TARGET_FRAMES,
    WINDOW_BEATS,
    AudioProcessor,
    LabelProcessor,
)


class BeatsDataset(Dataset):
    def __init__(self):
        self.data_dir = DATASET_DIR
        self.target_bpm = TARGET_BPM
        self.window_beats = WINDOW_BEATS
        self.fps = FPS
        self.target_frames = TARGET_FRAMES
        self.steps_per_beat = STEPS_PER_BEAT
        self.indecies: list[tuple[Path, int]] = []
        self.loader = MapLoader()
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()

        for meta, map_id_diff in self.loader.iter_processed_meta():
            num_windows = int(meta.total_beats / self.window_beats)
            map_dir = self.data_dir / map_id_diff
            self.indecies.extend(map(lambda i: (map_dir, i), range(num_windows)))

    def __len__(self):
        return len(self.indecies)

    def __getitem__(self, global_index: int):
        (map_dir, window_i) = self.indecies[global_index]

        meta = self.loader.load_meta(map_dir)

        audio_tensor: torch.Tensor = torch.load(map_dir / "features.pt")
        normalized_audio_tensor = self.audio_processor.normalize_audio_tensor(
            audio_tensor, meta.bpm, window_i
        )

        label_pair_tensor: torch.Tensor = torch.load(map_dir / "labels.pt")
        normalized_label_tensor = self.label_processor.normalize_label_tensor(
            label_pair_tensor, window_i
        )

        return (normalized_audio_tensor, normalized_label_tensor)
