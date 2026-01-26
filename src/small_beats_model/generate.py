import json
from pathlib import Path

import torch

from small_beats_model.loader import MODEL_PATH, PREDICTION_DIR
from small_beats_model.model import SmallBeatsNet
from small_beats_model.models import DiffFile, DiffNote
from small_beats_model.preprocessing import (
    STEPS_PER_BEAT,
    AudioProcessor,
    LabelProcessor,
)
from small_beats_model.utils import device_type
from small_beats_model.vocab import EMPTY_TOKEN, Vocab


class BeatGenerator:
    def __init__(self):
        self.device = torch.device(device_type)
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()
        self.model = SmallBeatsNet()
        self.vocab = Vocab()

        state_dict = torch.load(MODEL_PATH, map_location=device_type)
        self.model.load_state_dict(state_dict)
        self.model.to(device_type)
        self.model.eval()

        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    def infer(self, audio_path: Path):
        audio_tensor = self.audio_processor.process_audio(audio_path)
        bpm = self.audio_processor.get_bpm(audio_path)
        n_windows = self.audio_processor.get_audio_tensor_n_window(audio_tensor, bpm)

        generated_tokens: list[int] = []
        current_token = torch.tensor(
            [[EMPTY_TOKEN]], device=self.device, dtype=torch.long
        )
        hidden = None

        for window_i in range(n_windows):
            audio_slice = self.audio_processor.slice_audio_tensor(
                audio_tensor, bpm, window_i
            )
            audio_slice = audio_slice.unsqueeze(0).to(device_type)

            with torch.no_grad():
                audio_features = self.model.encode_audio(audio_slice)

            window_seq_len = audio_features.size(1)

            for step in range(window_seq_len):
                step_audio = audio_features[:, step : step + 1, :]
                logits, hidden = self.model.forward_rnn(
                    step_audio, current_token, hidden
                )
                probs = torch.softmax(logits[0, 0], dim=0)
                next_token_index = int(torch.argmax(probs).item())
                generated_tokens.append(next_token_index)
                current_token = torch.tensor([[next_token_index]], device=self.device)

        return generated_tokens

    def tokens_to_notes(self, predictions: list[int]):
        notes: list[DiffNote] = []

        for i, prediction in enumerate(predictions):
            if prediction == EMPTY_TOKEN:
                continue

            time = i / STEPS_PER_BEAT
            current_notes = self.vocab.decode(time=time, token=prediction)
            notes.extend(current_notes)

        return notes

    def save(self, output_dir_name: str, predictions: list[int], notes: list[DiffNote]):
        diff_file = DiffFile(_version="2.0.0", _notes=notes)

        output_dir = PREDICTION_DIR / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "tokens.json", "w") as f:
            f.write(json.dumps(predictions))

        with open(output_dir / "Expert.dat", "w") as f:
            f.write(json.dumps(diff_file.model_dump(by_alias=True), indent=2))

        return output_dir
