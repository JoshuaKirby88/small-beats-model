from math import ceil
from pathlib import Path

import librosa
import torch

from small_beats_model.models import DiffNote, VocabKey

SAMPLE_RATE = 22050
N_MFCC = 40
HOP_LENGTH = 512

STEPS_PER_BEAT = 4
NUM_COLORS = 2

NUM_DIRECTIONS = 9
NUM_COLS = 4
NUM_ROWS = 3
VOCAB_SIZE = NUM_DIRECTIONS * NUM_COLS * NUM_ROWS + 1


class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mfcc = N_MFCC
        self.hop_length = HOP_LENGTH

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

    def get_duration_s(self, audio_tensor: torch.Tensor):
        return audio_tensor.shape[1] * self.hop_length / self.sample_rate


class LabelProcessor:
    def __init__(self):
        self.steps_per_beat = STEPS_PER_BEAT
        self.num_colors = NUM_COLORS
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
        grid = torch.zeros(total_steps, 2, dtype=torch.long)

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


if __name__ == "__main__":
    audio_path = Path("data/raw/1e6ff/song.egg")
    audio_processor = AudioProcessor()
    tensor = audio_processor.process_audio(audio_path)
    print(f"Shape {tensor.shape}")
