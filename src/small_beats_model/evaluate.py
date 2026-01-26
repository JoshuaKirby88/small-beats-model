from math import log2

import torch

from small_beats_model.generate import BeatGenerator
from small_beats_model.loader import DATASET_DIR, MapLoader
from small_beats_model.models import DiffNote
from small_beats_model.preprocessing import STEPS_PER_BEAT, AudioProcessor
from small_beats_model.vocab import Vocab

N_GRAM_N = 2


class ModelEvaluator:
    def __init__(self):
        self.loader = MapLoader()
        self.generator = BeatGenerator()
        self.audio_processor = AudioProcessor()
        self.vocab = Vocab()

    def get_nps(self, bpm: float, predictions: list[int], notes: list[DiffNote]):
        duration_s = ((len(predictions) / STEPS_PER_BEAT) / bpm) * 60
        return len(notes) / duration_s

    def empty_ratio(self, tokens: list[int]):
        n_zeros = tokens.count(0)
        total = len(tokens)
        return n_zeros / total

    def note_density(self, tokens: list[int]):
        n_notes = len([t for t in tokens if t > 0])
        total = len(tokens)
        return n_notes / total

    def class_distribution(self, tokens: list[int]):
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        entropy = 0
        for count in counts.values():
            p = count / len(tokens)
            entropy += -p * log2(p)

        unique_tokens = len(counts)

        return {"unique_tokens": unique_tokens, "entropy": entropy}

    def pattern_diversity(self, tokens: list[int], n: int):
        n_grams: list[tuple[int, ...]] = []
        for i in range(len(tokens) - n + 1):
            n_gram = tokens[i : i + n]
            if any(t > 0 for t in n_gram):
                pattern = tuple(t for t in n_gram)
                n_grams.append(pattern)
        if len(n_grams) == 0:
            return 0.0
        else:
            unique = set(n_grams)
            return len(unique) / len(n_grams)

    def run_average(self, tokens: list[int]):
        empty_ratio = self.empty_ratio(tokens)
        note_density = self.note_density(tokens)
        class_distribution = self.class_distribution(tokens)
        pattern_diversity = self.pattern_diversity(tokens, N_GRAM_N)
        return {
            "empty_ratio": empty_ratio,
            "note_density": note_density,
            "unique_tokens": class_distribution["unique_tokens"],
            "entropy": class_distribution["entropy"],
            "pattern_diversity": pattern_diversity,
        }

    def run(self):
        tokens = []
        for _, _, map_id_diff in self.loader.iter_processed_meta():
            labels = torch.load(DATASET_DIR / map_id_diff / "labels.pt")
            tokens.extend(labels.tolist())
        print(self.run_average(tokens))

        for audio_path in self.loader.iter_eval_audio():
            bpm = self.audio_processor.get_bpm(audio_path)
            print(f"BPM: {bpm}")

            predictions = self.generator.infer(audio_path)
            notes = self.generator.tokens_to_notes(predictions)
            self.generator.save(audio_path.name, predictions, notes)
            print(f"NPS: {round(self.get_nps(bpm, predictions, notes), 2)}")
            print(self.run_average(predictions))


if __name__ == "__main__":
    model_eval = ModelEvaluator()
    model_eval.run()
