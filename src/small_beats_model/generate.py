import json
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F

from small_beats_model.dataset import FPS, TARGET_FRAMES, WINDOW_BEATS
from small_beats_model.loader import PREDICTION_DIR
from small_beats_model.model import SmallBeatsNet
from small_beats_model.models import DiffFile, DiffNote
from small_beats_model.preprocessing import (
    SAMPLE_RATE,
    STEPS_PER_BEAT,
    AudioProcessor,
    LabelProcessor,
)
from small_beats_model.train import MODEL_PATH

device = "mps" if torch.mps.is_available() else "cpu"


class BeatGenerator:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.window_beats = WINDOW_BEATS
        self.fps = FPS
        self.target_frames = TARGET_FRAMES
        self.step_per_beat = STEPS_PER_BEAT
        self.prediction_dir = PREDICTION_DIR
        self.model_path = MODEL_PATH

        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()
        self.model = SmallBeatsNet()
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def infer(self, audio_path: Path):
        audio_tensor = self.audio_processor.process_audio(audio_path)

        audio_array, _ = librosa.load(audio_path, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
        bpm = float(tempo)

        s_per_beat = 60 / bpm
        window_duration_s = s_per_beat * self.window_beats

        i = 0
        is_audio_end = False
        all_predictions: list[int] = []
        while not is_audio_end:
            start_time = window_duration_s * i
            end_time = start_time + window_duration_s

            audio_start_frame = int(start_time * self.fps)
            audio_end_frame = int(end_time * self.fps)

            audio_slice = audio_tensor[:, audio_start_frame:audio_end_frame]

            expected_audio_width = audio_end_frame - audio_start_frame
            current_audio_width = audio_slice.shape[1]
            if current_audio_width < expected_audio_width:
                is_audio_end = True
                pad_audio_amount = expected_audio_width - current_audio_width
                audio_slice = F.pad(audio_slice, (0, pad_audio_amount))

            audio_input = audio_slice.unsqueeze(0)
            audio_resampled = F.interpolate(
                input=audio_input,
                size=self.target_frames,
                mode="linear",
                align_corners=False,
            )

            final_audio = audio_resampled.to(device)
            logits = self.model(final_audio)
            predictions = torch.argmax(logits, dim=2).squeeze(0).tolist()
            all_predictions.extend(predictions)

            i += 1

        return all_predictions

    def save(self, output_dir_name: str, predictions: list[int]):
        id_to_key = self.label_processor.get_id_to_key()
        notes: list[DiffNote] = []

        for i, prediction in enumerate(predictions):
            if prediction == 0:
                continue
            id = id_to_key[prediction]
            time = i / self.step_per_beat
            note = DiffNote(
                _time=time,
                _lineIndex=id.col,
                _lineLayer=id.row,
                _type=id.color,
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
