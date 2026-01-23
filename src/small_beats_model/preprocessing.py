import math
from math import ceil
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F

from small_beats_model.models import DiffNote, VocabKey

STEPS_PER_BEAT = 4
NUM_COLORS = 2

NUM_DIRECTIONS = 9
NUM_COLS = 4
NUM_ROWS = 3
VOCAB_SIZE = NUM_DIRECTIONS * NUM_COLS * NUM_ROWS + 1

SAMPLE_RATE = 22050
N_MFCC = 40
HOP_LENGTH = 512

TARGET_BPM = 120
WINDOW_BEATS = 32
FPS = SAMPLE_RATE / HOP_LENGTH
_RAW_FRAMES = int(WINDOW_BEATS * (60 / TARGET_BPM) * FPS)
_OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
TARGET_FRAMES = ((_RAW_FRAMES + _OUTPUT_STEPS - 1) // _OUTPUT_STEPS) * _OUTPUT_STEPS


class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mfcc = N_MFCC
        self.hop_length = HOP_LENGTH
        self.window_beats = WINDOW_BEATS
        self.fps = FPS
        self.target_frames = TARGET_FRAMES

    def process_audio(self, audio_path: Path):
        (audio_array, _) = librosa.load(audio_path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(
            y=audio_array,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
        )
        tensor = torch.tensor(mfcc)
        return tensor

    def get_audio_tensor_n_window(self, audio_tensor: torch.Tensor, bpm: float):
        s_per_beat = 60 / bpm
        window_duration_s = s_per_beat * self.window_beats
        total_frames = audio_tensor.shape[1]
        frames_per_window = window_duration_s * self.fps
        n_windows = math.ceil(total_frames / frames_per_window)
        return n_windows

    def normalize_audio_tensor(
        self, audio_tensor: torch.Tensor, bpm: float, window_i: int
    ):
        s_per_beat = 60 / bpm
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

        return final_audio

    def get_bpm(self, audio_path: Path):
        audio_array, _ = librosa.load(audio_path, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
        return float(tempo)

    def get_duration_s(self, audio_tensor: torch.Tensor):
        return audio_tensor.shape[1] * self.hop_length / self.sample_rate


class LabelProcessor:
    def __init__(self):
        self.steps_per_beat = STEPS_PER_BEAT
        self.num_colors = NUM_COLORS
        self.window_beats = WINDOW_BEATS
        self.vocab = self.build_vocab()

    def build_vocab(self):
        index = 1  # 0=empty
        vocab: dict[VocabKey, int] = {}

        for direction in range(NUM_DIRECTIONS):
            for col in range(NUM_COLS):
                for row in range(NUM_ROWS):
                    key = VocabKey(direction=direction, col=col, row=row)
                    vocab[key] = index
                    index += 1

        return vocab

    def notes_to_grid(self, notes: list[DiffNote], total_beats: float):
        total_steps = ceil(total_beats * self.steps_per_beat)
        grid = torch.zeros(total_steps, 2, dtype=torch.uint8)

        for note in notes:
            step_index = int(round(note.time * self.steps_per_beat))
            if step_index >= total_steps:
                continue

            key = VocabKey(
                direction=note.cutDirection,
                col=note.lineIndex,
                row=note.lineLayer,
            )
            token = self.vocab.get(key, 0)
            grid[step_index, note.type] = token

        return grid

    def get_id_to_key(self) -> dict[int, VocabKey]:
        return {v: k for k, v in self.vocab.items()}

    def normalize_label_tensor(self, label_pair_tensor: torch.Tensor, window_i: int):
        label_start_frame = window_i * self.window_beats * self.steps_per_beat
        label_end_frame = label_start_frame + self.window_beats * self.steps_per_beat
        label_pair_slice = label_pair_tensor[label_start_frame:label_end_frame]

        expected_label_width = label_end_frame - label_start_frame
        current_label_width = label_pair_slice.shape[0]
        if current_label_width < expected_label_width:
            pad_label_amount = expected_label_width - current_label_width
            label_pair_slice = F.pad(label_pair_slice, (0, 0, 0, pad_label_amount))

        label_flat_slice = label_pair_slice.flatten()
        label_flat_slice_long = label_flat_slice.long()
        return label_flat_slice_long


if __name__ == "__main__":
    audio_path = Path("data/raw/1e6ff/song.egg")
    audio_processor = AudioProcessor()
    tensor = audio_processor.process_audio(audio_path)
    print(f"Shape {tensor.shape}")
