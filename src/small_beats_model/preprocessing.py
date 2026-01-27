import math
import warnings
from math import ceil
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from small_beats_model.models import DiffNote
from small_beats_model.vocab import Vocab

warnings.filterwarnings(
    "ignore",
    message=r".*librosa\.core\.audio\.__audioread_load.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore", message=".*PySoundFile failed.*", category=UserWarning
)

NUM_COLORS = 2
NUM_DIRECTIONS = 9
NUM_COLS = 4
NUM_ROWS = 3

SAMPLE_RATE = 22050
N_MFCC = 40
HOP_LENGTH = 512

STEPS_PER_BEAT = 4
TARGET_BPM = 120
WINDOW_BEATS = 32
FPS = SAMPLE_RATE / HOP_LENGTH
_RAW_FRAMES = int(WINDOW_BEATS * (60 / TARGET_BPM) * FPS)
_OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
TARGET_FRAMES = ((_RAW_FRAMES + _OUTPUT_STEPS - 1) // _OUTPUT_STEPS) * _OUTPUT_STEPS


class AudioProcessor:
    def process_audio(self, audio_path: Path, song_offset=0.0):
        (audio_array, _) = librosa.load(
            audio_path, sr=SAMPLE_RATE, offset=max(0, song_offset)
        )

        if song_offset < 0:
            silence_s = abs(song_offset)
            silence_samples = int(silence_s * SAMPLE_RATE)
            audio_array = np.pad(audio_array, (silence_samples, 0), mode="constant")

        mfcc = librosa.feature.mfcc(
            y=audio_array,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
        )

        tensor = torch.tensor(mfcc)
        return tensor

    def slice_audio_tensor(self, audio_tensor: torch.Tensor, bpm: float, window_i: int):
        s_per_beat = 60 / bpm
        window_duration_s = s_per_beat * WINDOW_BEATS

        start_time = window_duration_s * window_i
        end_time = start_time + window_duration_s
        audio_start_frame = int(start_time * FPS)
        audio_end_frame = int(end_time * FPS)

        audio_slice = audio_tensor[:, audio_start_frame:audio_end_frame]

        expected_audio_width = audio_end_frame - audio_start_frame
        current_audio_width = audio_slice.shape[1]
        if current_audio_width < expected_audio_width:
            pad_audio_amount = expected_audio_width - current_audio_width
            audio_slice = F.pad(audio_slice, (0, pad_audio_amount))

        audio_input = audio_slice.unsqueeze(0)
        audio_resampled = F.interpolate(
            input=audio_input,
            size=TARGET_FRAMES,
            mode="linear",
            align_corners=False,
        )
        return audio_resampled.squeeze(0)

    def get_audio_tensor_n_window(self, audio_tensor: torch.Tensor, bpm: float):
        s_per_beat = 60 / bpm
        window_duration_s = s_per_beat * WINDOW_BEATS
        total_frames = audio_tensor.shape[1]
        frames_per_window = window_duration_s * FPS
        n_windows = math.ceil(total_frames / frames_per_window)
        return n_windows

    def get_bpm(self, audio_path: Path):
        audio_array, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        tempo, _ = librosa.beat.beat_track(y=audio_array, sr=SAMPLE_RATE)
        return float(tempo)

    def get_duration_s(self, audio_tensor: torch.Tensor):
        return audio_tensor.shape[1] * HOP_LENGTH / SAMPLE_RATE

    def get_audio_steps(self, audio_tensor: torch.Tensor, bpm: float):
        duration_s = self.get_duration_s(audio_tensor)
        total_beats = duration_s * (bpm / 60)
        total_steps = total_beats * STEPS_PER_BEAT
        return total_steps


class LabelProcessor:
    def __init__(self):
        self.vocab = Vocab()

    def notes_to_grid(self, notes: list[DiffNote], total_beats: float):
        total_steps = ceil(total_beats * STEPS_PER_BEAT)
        grid = torch.zeros(total_steps, dtype=torch.long)

        for time, current_notes in self.vocab.group_notes_by_time(notes):
            step_index = int(round(time * STEPS_PER_BEAT))
            if step_index >= total_steps:
                continue
            token = self.vocab.encode(list(current_notes))
            grid[step_index] = token

        return grid

    def slice_label_tensor(self, label_tensor: torch.Tensor, window_i: int):
        label_start_frame = window_i * WINDOW_BEATS * STEPS_PER_BEAT
        label_end_frame = label_start_frame + WINDOW_BEATS * STEPS_PER_BEAT
        label_slice = label_tensor[label_start_frame:label_end_frame]

        expected_label_width = label_end_frame - label_start_frame
        current_label_width = label_slice.shape[0]
        if current_label_width < expected_label_width:
            pad_label_amount = expected_label_width - current_label_width
            label_slice = F.pad(label_slice, (0, pad_label_amount))

        return label_slice


if __name__ == "__main__":
    audio_path = Path("data/raw/1e6ff/song.egg")
    audio_processor = AudioProcessor()
    tensor = audio_processor.process_audio(audio_path)
    print(f"Shape {tensor.shape}")
