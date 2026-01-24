import json
from pathlib import Path

import torch

from small_beats_model.loader import MODEL_PATH, PREDICTION_DIR
from small_beats_model.model import SmallBeatsNet
from small_beats_model.models import DiffFile, DiffNote
from small_beats_model.preprocessing import (
    NUM_COLORS,
    STEPS_PER_BEAT,
    AudioProcessor,
    LabelProcessor,
)
from small_beats_model.utils import device_type


class BeatGenerator:
    def __init__(self):
        self.device = torch.device(device_type)
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()
        self.model = SmallBeatsNet()

        state_dict = torch.load(MODEL_PATH, map_location=device_type)
        self.model.load_state_dict(state_dict)
        self.model.to(device_type)
        self.model.eval()

        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    def infer(self, audio_path: Path):
        audio_tensor = self.audio_processor.process_audio(audio_path)
        bpm = self.audio_processor.get_bpm(audio_path)
        n_windows = self.audio_processor.get_audio_tensor_n_window(audio_tensor, bpm)

        generated_tokens = []
        current_token = torch.tensor([0], device=self.device, dtype=torch.long)
        hidden = None

        for window_i in range(n_windows):
            normalized_audio = self.audio_processor.normalize_and_slice_audio_tensor(
                audio_tensor, bpm, window_i
            )
            normalized_audio = normalized_audio.unsqueeze(0).to(device_type)

            with torch.no_grad():
                audio_features = self.model.encode_audio(normalized_audio)

            window_seq_len = audio_features.size(1)

            for step in range(window_seq_len):
                step_audio = audio_features[:, step : step + 1, :]
                color_id = torch.tensor(
                    [[step % NUM_COLORS]], device=self.device, dtype=torch.long
                )
                logits, hidden = self.model.forward_rnn(
                    step_audio, current_token, color_id, hidden
                )
                probs = torch.softmax(logits[0, 0], dim=0)
                next_token_index = torch.argmax(probs).item()
                generated_tokens.append(next_token_index)
                current_token = torch.tensor([next_token_index], device=self.device)

        return generated_tokens

    def save(self, output_dir_name: str, predictions: list[int]):
        id_to_key = self.label_processor.get_id_to_key()
        notes: list[DiffNote] = []

        for i in range(int(len(predictions) / 2)):
            time = i / STEPS_PER_BEAT
            for color in range(NUM_COLORS):
                prediction = predictions[i * 2 + color]
                if prediction == 0:
                    continue

                id = id_to_key[prediction]
                time = i / STEPS_PER_BEAT
                note = DiffNote(
                    _time=time,
                    _lineIndex=id.col,
                    _lineLayer=id.row,
                    _type=color,
                    _cutDirection=id.direction,
                )
                notes.append(note)

        diff_file = DiffFile(_version="2.0.0", _notes=notes)

        output_dir = PREDICTION_DIR / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "tokens.json", "w") as f:
            f.write(json.dumps(predictions))

        with open(output_dir / "Expert.dat", "w") as f:
            f.write(json.dumps(diff_file.model_dump(by_alias=True), indent=2))

        return output_dir
