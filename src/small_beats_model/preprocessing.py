from math import ceil
from pathlib import Path
from typing import Dict

import librosa
import torch

from small_beats_model.models import MapDiffNote, VocabKey

SAMPLE_RATE = 22050
N_MFCC = 40
HOP_LENGTH = 512


class AudioProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    def process_audio(self, audio_path: Path) -> torch.Tensor:
        (audio_array, sample_rate) = librosa.load(audio_path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(
            y=audio_array,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
        )
        tensor = torch.tensor(mfcc)
        return tensor

    def get_duration_s(self, audio_tensor: torch.Tensor):
        return audio_tensor.shape[1] * self.hop_length / self.sample_rate


STEPS_PER_BEAT = 4


class LabelProcessor:
    def __init__(self, steps_per_beat=STEPS_PER_BEAT):
        self.steps_per_beat = steps_per_beat
        self.vocab = self.build_vocab()

    def build_vocab(self) -> Dict[VocabKey, int]:
        index = 1  # 0=empty
        vocab: Dict[VocabKey, int] = {}

        for color in [0, 1]:
            for direction in range(9):
                for col in range(4):
                    for row in range(3):
                        key = VocabKey(
                            color=color, direction=direction, col=col, row=row
                        )
                        vocab[key] = index
                        index += 1

        return vocab

    def notes_to_grid(
        self, notes: list[MapDiffNote], total_beats: float
    ) -> torch.Tensor:
        total_steps = ceil(total_beats * self.steps_per_beat)
        grid = torch.zeros(total_steps, dtype=torch.long)

        for note in notes:
            step_index = int(round(note.time * self.steps_per_beat))
            if step_index >= total_steps:
                continue
            key = VocabKey(
                color=note.type,
                direction=note.cutDirection,
                col=note.lineIndex,
                row=note.lineLayer,
            )
            token = self.vocab.get(key, 0)
            grid[step_index] = token

        return grid


if __name__ == "__main__":
    audio_path = Path("data/raw/1e6ff/song.egg")
    audio_processor = AudioProcessor()
    tensor = audio_processor.process_audio(audio_path)
    print(f"Shape {tensor.shape}")
