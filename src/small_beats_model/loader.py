import json
from pathlib import Path

from small_beats_model.models import DatasetMeta, DiffFile, InfoDiff, InfoFile

SCRAPED_DATA_DIR = Path("data/raw")
DATASET_DIR = Path("data/processed")
PREDICTION_DIR = Path("data/predictions")


class MapLoader:
    def __init__(self):
        self.scraped_dir = SCRAPED_DATA_DIR
        self.dataset_dir = DATASET_DIR
        self.prediction_dir = PREDICTION_DIR

    def get_audio_path(self, map_id: str):
        info_file = self.load_info_file(self.scraped_dir / map_id)
        return self.scraped_dir / map_id / info_file.songFilename

    def load_info_file(self, map_dir: Path):
        info_path = map_dir / "Info.dat"
        with open(info_path, "r", encoding="utf-8") as f:
            return InfoFile.model_validate(json.load(f))

    def load_diff_file(self, diff_path: Path):
        try:
            with open(diff_path, "r", encoding="utf-8") as f:
                return DiffFile.model_validate(json.load(f))
        except FileNotFoundError:
            return None

    def load_diff_files(self, map_dir: Path, info_file: InfoFile):
        diff_tuples: list[tuple[InfoDiff, DiffFile]] = []
        for diff_set in info_file.difficultyBeatmapSets:
            for diff_map in diff_set.difficultyBeatmaps:
                diff_path = map_dir / diff_map.beatmapFilename
                maybe_diff_file = self.load_diff_file(diff_path)
                if maybe_diff_file is not None:
                    diff_tuples.append((diff_map, maybe_diff_file))
        return diff_tuples

    def iter_scraped(self):
        for map_dir in self.scraped_dir.iterdir():
            if not map_dir.is_dir():
                continue
            info_file = self.load_info_file(map_dir)
            diff_tuples = self.load_diff_files(map_dir, info_file)
            map_id = map_dir.name
            yield (info_file, diff_tuples, map_id)

    def load_meta(self, map_dir: Path):
        with open(map_dir / "meta.json", "r", encoding="utf-8") as f:
            return DatasetMeta.model_validate(json.load(f))

    def iter_processed_meta(self):
        for map_dir in self.dataset_dir.iterdir():
            if not map_dir.is_dir():
                continue
            meta = self.load_meta(map_dir)
            map_id_diff = map_dir.name
            yield (meta, map_id_diff)

    def iter_processed_meta_by_map_id(self, map_id: str):
        dirs = [p for p in self.dataset_dir.iterdir() if p.name.startswith(map_id)]
        for map_dir in dirs:
            if not map_dir.is_dir():
                continue
            meta = self.load_meta(map_dir)
            map_id_diff = map_dir.name
            yield (meta, map_id_diff)

    def load_prediction_tokens(self, dir: Path) -> list[int]:
        token_path = dir / "tokens.json"
        with open(token_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_prediction_diff_map(self, dir: Path):
        diff_path = dir / "Expert.dat"
        with open(diff_path) as f:
            return DiffFile.model_validate(json.load(f))

    def iter_predicted(self):
        for dir in self.prediction_dir.iterdir():
            if not dir.is_dir():
                continue
            tokens = self.load_prediction_tokens(dir)
            diff_file = self.load_prediction_diff_map(dir)
            map_id = dir.name
            yield (tokens, diff_file, map_id)
