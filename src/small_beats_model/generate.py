import json
from pathlib import Path

import torch

from small_beats_model.loader import MODEL_PATH, PREDICTION_DIR
from small_beats_model.model import OUTPUT_STEPS, SmallBeatsNet
from small_beats_model.models import DiffFile, DiffNote
from small_beats_model.preprocessing import (
    STEPS_PER_BEAT,
    AudioProcessor,
    LabelProcessor,
)
from small_beats_model.utils import device_type
from small_beats_model.vocab import EMPTY_TOKEN, Vocab

TEMPERATURE = 1
TOP_K = 20


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

    def sample_next_token(self, logits: torch.Tensor):
        scaled_logits = logits / TEMPERATURE
        top_values, top_indices = torch.topk(scaled_logits, TOP_K)
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits[top_indices] = top_values
        probs = torch.softmax(filtered_logits, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        return int(sampled_index.item())

    def infer(self, audio_path: Path):
        audio_tensor = self.audio_processor.process_audio(audio_path)
        bpm = self.audio_processor.get_bpm(audio_path)
        n_windows = self.audio_processor.get_audio_tensor_n_window(audio_tensor, bpm)
        total_steps = self.audio_processor.get_audio_steps(audio_tensor, bpm)

        all_audio_features = []
        for window_i in range(n_windows):
            audio_slice = self.audio_processor.slice_audio_tensor(
                audio_tensor, bpm, window_i
            )
            audio_slice = audio_slice.unsqueeze(0).to(device_type)
            with torch.no_grad():
                features = self.model.encode_audio(audio_slice)
            all_audio_features.append(features)
        all_audio_features = torch.cat(all_audio_features, dim=1)

        audio_dim = all_audio_features.size(-1)
        audio_buffer = torch.zeros(1, 128, audio_dim, device=self.device)
        generated_tokens: list[int] = []
        token_buffer = [EMPTY_TOKEN] * OUTPUT_STEPS

        for window_i in range(n_windows):
            audio_slice = self.audio_processor.slice_audio_tensor(
                audio_tensor, bpm, window_i
            )
            audio_slice = audio_slice.unsqueeze(0).to(device_type)

            with torch.no_grad():
                audio_features = self.model.encode_audio(audio_slice)

            window_seq_len = audio_features.size(1)

            for step in range(window_seq_len):
                global_step = (window_i * window_seq_len) + step
                if global_step > total_steps:
                    break

                current_audio = all_audio_features[:, global_step : global_step + 1, :]
                audio_buffer = torch.cat([audio_buffer[:, 1:, :], current_audio], dim=1)
                prev_tokens = torch.tensor([token_buffer], device=self.device)
                logits, _ = self.model.forward_rnn(
                    audio_features=audio_buffer,
                    prev_tokens=prev_tokens,
                    hidden=None,
                    return_all=False,
                )
                next_token_index = self.sample_next_token(logits[0])
                generated_tokens.append(next_token_index)
                token_buffer = token_buffer[1:] + [next_token_index]

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
