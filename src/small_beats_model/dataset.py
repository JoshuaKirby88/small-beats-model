import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from small_beats_model.models import DatasetMeta
from small_beats_model.preprocessing import HOP_LENGTH, SAMPLE_RATE, STEPS_PER_BEAT

DATA_DIR = Path("data/processed")
TARGET_BPM = 120
WINDOW_BEATS = 32
FPS = SAMPLE_RATE / HOP_LENGTH
_RAW_FRAMES = int(WINDOW_BEATS * (60 / TARGET_BPM) * FPS)
_OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
TARGET_FRAMES = ((_RAW_FRAMES + _OUTPUT_STEPS - 1) // _OUTPUT_STEPS) * _OUTPUT_STEPS


class BeatsDataset(Dataset):
    def __init__(
        self,
        data_dir=DATA_DIR,
        target_bpm=TARGET_BPM,
        window_beats=WINDOW_BEATS,
        fps=FPS,
        target_frames=TARGET_FRAMES,
        steps_per_beat=STEPS_PER_BEAT,
    ):
        self.data_dir = data_dir
        self.target_bpm = target_bpm
        self.window_beats = window_beats
        self.fps = fps
        self.target_frames = target_frames
        self.steps_per_beat = steps_per_beat
        self.indecies: list[tuple[Path, int]] = []

        for map_dir in self.data_dir.iterdir():
            if not map_dir.is_dir():
                continue

            with open(map_dir / "meta.json", "r") as f:
                meta = DatasetMeta.model_validate(json.load(f))

            num_windows = int(meta.total_beats / self.window_beats)
            self.indecies.extend(
                map(lambda window_i: (map_dir, window_i), range(num_windows))
            )

    def __len__(self):
        return len(self.indecies)

    def __getitem__(self, index: int):
        (map_dir, window_i) = self.indecies[index]

        with open(map_dir / "meta.json") as f:
            meta = DatasetMeta.model_validate(json.load(f))

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

        label_tensor: torch.Tensor = torch.load(map_dir / "labels.pt")

        label_start_frame = window_i * self.window_beats * self.steps_per_beat
        label_end_frame = label_start_frame + self.window_beats * self.steps_per_beat
        label_slice = label_tensor[label_start_frame:label_end_frame]

        expected_label_width = label_end_frame - label_start_frame
        current_label_width = label_slice.shape[0]
        if current_label_width < expected_label_width:
            pad_label_amount = expected_label_width - current_label_width
            label_slice = F.pad(label_slice, (0, pad_label_amount))

        return (final_audio, label_slice)
