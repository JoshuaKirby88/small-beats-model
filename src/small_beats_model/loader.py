import json
from pathlib import Path
from typing import Optional

from small_beats_model.models import (
    MapDiffFile,
    MapInfoDiff,
    MapInfoDiffSet,
    MapInfoFile,
)

CHARACTERISTIC_FILTERS = ["Standard"]


class MapLoader:
    def __init__(self, characteristic_filters=CHARACTERISTIC_FILTERS):
        self.characteristic_filters = characteristic_filters

    def is_v2_info(self, raw_info: dict) -> bool:
        return raw_info.get("_version", "").startswith("2.")

    def is_valid_characteristic_diff(self, diff_set: MapInfoDiffSet) -> bool:
        return diff_set.beatmapCharacteristicName in self.characteristic_filters

    def is_v2_diff(self, raw_diff_map: dict) -> bool:
        return "_notes" in raw_diff_map and len(raw_diff_map.get("_notes", [])) > 0

    def load_info(self, path: Path) -> Optional[MapInfoFile]:
        with path.open("r", encoding="utf-8") as f:
            raw_info = json.load(f)
        if self.is_v2_info(raw_info):
            return MapInfoFile.model_validate(raw_info)

    def get_valid_diff_sets(self, info: MapInfoFile):
        return filter(
            lambda diff_set: diff_set.beatmapCharacteristicName
            in self.characteristic_filters,
            info.difficultyBeatmapSets,
        )

    def load_difficulties(self, map_dir: Path, diff_set: MapInfoDiffSet):
        diff_map_tuples: list[tuple[MapInfoDiff, MapDiffFile]] = []

        for diff_map in diff_set.difficultyBeatmaps:
            diff_file = map_dir / diff_map.beatmapFilename
            with diff_file.open("r", encoding="utf-8") as f:
                raw_diff_map = json.load(f)
            if self.is_v2_diff(raw_diff_map):
                diff_map_tuples.append(
                    (diff_map, MapDiffFile.model_validate(raw_diff_map))
                )

        return diff_map_tuples

    def load_map(self, map_dir: Path):
        if not map_dir.is_dir():
            return

        info = self.load_info(map_dir / "Info.dat")
        if info is None:
            return

        diff_sets = self.get_valid_diff_sets(info)

        diff_tuples: list[tuple[MapInfoDiff, MapDiffFile]] = []
        for diff_set in diff_sets:
            diff_tuples.extend(self.load_difficulties(map_dir, diff_set))

        return (info, diff_tuples)
