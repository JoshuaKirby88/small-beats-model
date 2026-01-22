import json
from math import log2

import torch

from small_beats_model.generate import PREDICTION_DIR
from small_beats_model.loader import DATASET_DIR, SCRAPED_DATA_DIR, MapLoader
from small_beats_model.models import DatasetMeta

N_GRAM_N = 2


class MapEvaluator:
    def __init__(self):
        self.scraped_data_dir = SCRAPED_DATA_DIR
        self.dataset_dir = DATASET_DIR
        self.predicted_beats_dir = PREDICTION_DIR
        self.n_gram_n = N_GRAM_N
        self.metrics = [
            "empty_ratio",
            "note_density",
            "unique_tokens",
            "entropy",
            "pattern_diversity",
        ]
        self.loader = MapLoader()

    def load_truths(self, map_id: str):
        truths: list[tuple[DatasetMeta, list[int], str]] = []
        for meta, map_id_diff in self.loader.iter_processed_meta_by_map_id(map_id):
            label_path = self.dataset_dir / map_id_diff / "labels.pt"
            label_tensor: torch.Tensor = torch.load(label_path)
            token_pairs: list[list[int]] = label_tensor.tolist()
            tokens = [t for t_pair in token_pairs for t in t_pair]
            truths.append((meta, tokens, map_id_diff))
        return truths

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

    def run_average(self, tokens_list: list[list[int]]):
        flat_tokens = [t for ts in tokens_list for t in ts]

        empty_ratio = self.empty_ratio(flat_tokens)
        note_density = self.note_density(flat_tokens)
        class_distribution = self.class_distribution(flat_tokens)
        pattern_diversity = self.pattern_diversity(flat_tokens, self.n_gram_n)

        return {
            "empty_ratio": empty_ratio,
            "note_density": note_density,
            "unique_tokens": class_distribution["unique_tokens"],
            "entropy": class_distribution["entropy"],
            "pattern_diversity": pattern_diversity,
        }

    def run(self):
        results: list[dict[str, dict[str, float]]] = []
        for tokens, diff_file, map_id in self.loader.iter_predicted():
            truths = self.load_truths(map_id)
            true_tokens_list = [tokens for (_, tokens, _) in truths]
            result = {
                "true": self.run_average(true_tokens_list),
                "prediction": self.run_average([tokens]),
            }
            results.append(result)
        return results

    def run_aggregate(self):
        results = self.run()
        average_diffs: dict[str, float] = {}
        for metric in self.metrics:
            diff = (r["prediction"][metric] - r["true"][metric] for r in results)
            average_diffs[metric] = sum(diff) / len(results)
        return average_diffs


if __name__ == "__main__":
    eval = MapEvaluator()
    average_diffs = eval.run_aggregate()
    print(json.dumps(average_diffs, indent=2))
