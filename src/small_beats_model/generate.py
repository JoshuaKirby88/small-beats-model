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
from src.small_beats_model.utils import device_type


class BeatGenerator:
    def __init__(self):
        self.step_per_beat = STEPS_PER_BEAT
        self.prediction_dir = PREDICTION_DIR
        self.num_colors = NUM_COLORS
        self.device_type = device_type
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()
        self.model = SmallBeatsNet()

        state_dict = torch.load(MODEL_PATH, map_location=self.device_type)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device_type)
        self.model.eval()

        self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def infer(self, audio_path: Path):
        audio_tensor = self.audio_processor.process_audio(audio_path)
        bpm = self.audio_processor.get_bpm(audio_path)
        n_windows = self.audio_processor.get_audio_tensor_n_window(audio_tensor, bpm)

        all_predictions: list[int] = []

        for window_i in range(n_windows):
            normalized_audio_tensor = self.audio_processor.normalize_audio_tensor(
                audio_tensor, bpm, window_i
            )
            normalized_audio_tensor = normalized_audio_tensor.unsqueeze(0).to(
                self.device_type
            )

            logits = self.model(normalized_audio_tensor)
            predictions = torch.argmax(logits, dim=2).squeeze(0).tolist()
            all_predictions.extend(predictions)

        return all_predictions

    def save(self, output_dir_name: str, predictions: list[int]):
        id_to_key = self.label_processor.get_id_to_key()
        notes: list[DiffNote] = []

        for i in range(int(len(predictions) / 2)):
            time = i / self.step_per_beat
            for color in range(self.num_colors):
                prediction = predictions[i * 2 + color]
                if prediction == 0:
                    continue

                id = id_to_key[prediction]
                time = i / self.step_per_beat
                note = DiffNote(
                    _time=time,
                    _lineIndex=id.col,
                    _lineLayer=id.row,
                    _type=color,
                    _cutDirection=id.direction,
                )
                notes.append(note)

        diff_file = DiffFile(_version="2.0.0", _notes=notes)

        output_dir = self.prediction_dir / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "tokens.json", "w") as f:
            f.write(json.dumps(predictions))

        with open(output_dir / "Expert.dat", "w") as f:
            f.write(json.dumps(diff_file.model_dump(by_alias=True), indent=2))

        return output_dir
