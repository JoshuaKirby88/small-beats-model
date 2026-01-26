from math import log2
from pathlib import Path

import torch

from small_beats_model.generate import BeatGenerator
from small_beats_model.loader import DATASET_DIR, MapLoader
from small_beats_model.package import Packager
from small_beats_model.preprocessing import NUM_COLORS, STEPS_PER_BEAT, AudioProcessor
from small_beats_model.vocab import EMPTY_TOKEN, Vocab

N_GRAM_N = 2


class ModelEvaluator:
    def __init__(self):
        self.loader = MapLoader()
        self.generator = BeatGenerator()
        self.packager = Packager()
        self.audio_processor = AudioProcessor()
        self.vocab = Vocab()

    def get_nps(self, bpm: float, predictions: list[int]):
        duration_s = ((len(predictions) / STEPS_PER_BEAT) / bpm) * 60
        notes = [p for p in predictions if p != EMPTY_TOKEN]
        return len(notes) / duration_s

    def empty_ratio(self, tokens: list[int]):
        n_zeros = tokens.count(0)
        total = len(tokens)
        return n_zeros / total

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

    def pattern_diversity(self, tokens: list[int]):
        n_grams: list[tuple[int, ...]] = []
        for i in range(len(tokens) - N_GRAM_N + 1):
            n_gram = tokens[i : i + N_GRAM_N]
            if any(t > 0 for t in n_gram):
                pattern = tuple(t for t in n_gram)
                n_grams.append(pattern)
        if len(n_grams) == 0:
            return 0.0
        else:
            unique = set(n_grams)
            return len(unique) / len(n_grams)

    def double_cut(self, tokens: list[int]):
        count = 0
        notes = [
            n
            for (i, t) in enumerate(tokens)
            for n in self.vocab.decode(i / STEPS_PER_BEAT, t)
        ]
        for color in range(NUM_COLORS):
            color_notes = [n for n in notes if n.type == color]
            for a, b in zip(color_notes[:-1], color_notes[1:]):
                if a.cutDirection == b.cutDirection:
                    count += 1
        return count

    def run_on_song(self, bpm: float, tokens: list[int]):
        nps = self.get_nps(bpm, tokens)
        empty_ratio = self.empty_ratio(tokens)
        class_distribution = self.class_distribution(tokens)
        pattern_diversity = self.pattern_diversity(tokens)
        double_cut_count = self.double_cut(tokens)
        return {
            "nps": nps,
            "empty_ratio": empty_ratio,
            "unique_tokens": class_distribution["unique_tokens"],
            "entropy": class_distribution["entropy"],
            "pattern_diversity": pattern_diversity,
            "double_cut_count": double_cut_count,
        }

    def get_average(self, songs: list[tuple[float, list[int]]]):
        result = {}
        results = [self.run_on_song(bpm, tokens) for (bpm, tokens) in songs]
        keys = results[0].keys()
        for key in keys:
            values = [r[key] for r in results]
            average = sum(values) / len(results)
            result[key] = average
        return result

    def run(self):
        songs = [
            (meta.bpm, torch.load(DATASET_DIR / map_id_diff / "labels.pt").tolist())
            for meta, _, map_id_diff in self.loader.iter_processed_meta()
        ]
        print(self.get_average(songs))

        for audio_path in self.loader.iter_eval_audio():
            bpm = self.audio_processor.get_bpm(audio_path)
            print(f"BPM: {bpm}")

            predictions = self.generator.infer(audio_path)
            notes = self.generator.tokens_to_notes(predictions)
            output_dir = self.generator.save(audio_path.stem, predictions, notes)
            export_path = self.packager.package(
                diff_path=output_dir / "Expert.dat",
                audio_path=audio_path,
                cover_path=Path(""),
                song_name=audio_path.stem,
                song_author_name="Temp",
                author_name="Josh",
            )
            print(export_path.stem)
            print(self.run_on_song(bpm, predictions))


if __name__ == "__main__":
    model_eval = ModelEvaluator()
    model_eval.run()
