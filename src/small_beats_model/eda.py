import json
from pathlib import Path

from small_beats_model.models import MapDiffFile, MapInfoFile

DATA_DIR = Path("data/raw")
INFO_FILE_NAME = "Info.dat"
CHARACTERISTIC_FILTER = "Standard"


class BeatSaverEDA:
    def __init__(self):
        print()

    def analyze_difficulty(self, diff: MapDiffFile, info: MapInfoFile):
        last_beat_time = diff.notes[-1].time
        bpm = info.beatsPerMinute
        last_time_s = (last_beat_time / bpm) * 60

        total_notes = len(diff.notes)
        nps = total_notes / last_time_s

        layer_counts = {0: 0, 1: 0, 2: 0}
        for n in diff.notes:
            layer_counts[n.lineLayer] += 1

        return last_time_s, nps, layer_counts

    def run(
        self,
        data_dir=DATA_DIR,
        info_file_name=INFO_FILE_NAME,
        characteristic_filter=CHARACTERISTIC_FILTER,
    ):
        for map_dir in data_dir.iterdir():
            print(f"Processing map directory: {map_dir}")
            if not map_dir.is_dir():
                continue

            info_file = map_dir / info_file_name
            with info_file.open("r", encoding="utf-8") as f:
                raw_info = json.load(f)
                if not raw_info.get("_version").startswith("2."):
                    continue
                info = MapInfoFile.model_validate(raw_info)

            for diff_set in info.difficultyBeatmapSets:
                if diff_set.beatmapCharacteristicName != characteristic_filter:
                    continue

                for diff_map in diff_set.difficultyBeatmaps:
                    diff_file = map_dir / diff_map.beatmapFilename
                    with diff_file.open("r", encoding="utf-8") as f:
                        raw_diff_map = json.load(f)
                        if (
                            "_notes" not in raw_diff_map
                            or len(raw_diff_map.get("_notes")) == 0
                        ):
                            continue
                        diff_map = MapDiffFile.model_validate(raw_diff_map)

                    stats = self.analyze_difficulty(diff_map, info)
                    print(
                        f"[{info.songName}] NPS: {stats[1]:.2f} | Duration: {stats[0]:.0f}s | Layer Counts: {stats[2]}"
                    )


if __name__ == "__main__":
    eda = BeatSaverEDA()
    eda.run()
