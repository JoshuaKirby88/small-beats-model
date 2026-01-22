from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from small_beats_model.build_dataset import DATASET_DIR
from small_beats_model.loader import MapLoader
from small_beats_model.preprocessing import (
    HOP_LENGTH,
    SAMPLE_RATE,
    STEPS_PER_BEAT,
)

TARGET_BPM = 120
WINDOW_BEATS = 32
FPS = SAMPLE_RATE / HOP_LENGTH
_RAW_FRAMES = int(WINDOW_BEATS * (60 / TARGET_BPM) * FPS)
_OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
TARGET_FRAMES = ((_RAW_FRAMES + _OUTPUT_STEPS - 1) // _OUTPUT_STEPS) * _OUTPUT_STEPS


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

        for meta, map_id_diff in self.loader.iter_processed_meta():
            num_windows = int(meta.total_beats / self.window_beats)
            map_dir = self.data_dir / map_id_diff
            self.indecies.extend(map(lambda i: (map_dir, i), range(num_windows)))

    def __len__(self):
        return len(self.indecies)

    def __getitem__(self, index: int):
        (map_dir, window_i) = self.indecies[index]

        meta = self.loader.load_meta(map_dir)
        audio_tensor: torch.Tensor = torch.load(map_dir / "features.pt")

        s_per_beat = 60 / meta.bpm
        window_duration_s = s_per_beat * self.window_beats
        start_time = window_duration_s * window_i
        end_time = start_time + window_duration_s

        audio_start_frame = int(start_time * self.fps)
        audio_end_frame = int(end_time * self.fps)

        audio_slice = audio_tensor[:, audio_start_frame:audio_end_frame]

        expected_audio_width = audio_end_frame - audio_start_frame
        current_audio_width = audio_slice.shape[1]
        if current_audio_width < expected_audio_width:
            pad_audio_amount = expected_audio_width - current_audio_width
            audio_slice = F.pad(audio_slice, (0, pad_audio_amount))

        audio_input = audio_slice.unsqueeze(0)
        audio_resampled = F.interpolate(
            input=audio_input,
            size=self.target_frames,
            mode="linear",
            align_corners=False,
        )
        final_audio = audio_resampled.squeeze(0)

        label_pair_tensor: torch.Tensor = torch.load(map_dir / "labels.pt")

        label_start_frame = window_i * self.window_beats * self.steps_per_beat
        label_end_frame = label_start_frame + self.window_beats * self.steps_per_beat
        label_pair_slice = label_pair_tensor[label_start_frame:label_end_frame]

        expected_label_width = label_end_frame - label_start_frame
        current_label_width = label_pair_slice.shape[0]
        if current_label_width < expected_label_width:
            pad_label_amount = expected_label_width - current_label_width
            label_pair_slice = F.pad(label_pair_slice, (0, 0, 0, pad_label_amount))

        label_flat_slice = label_pair_slice.flatten()

        return (final_audio, label_flat_slice)
