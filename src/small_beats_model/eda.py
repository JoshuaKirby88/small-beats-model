from pathlib import Path

from small_beats_model.loader import MapLoader
from small_beats_model.models import MapDiffFile, MapInfoFile

DATA_DIR = Path("data/raw")


class BeatSaverEDA:
    def __init__(self):
        self.loader = MapLoader()

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

    def run(self, data_dir=DATA_DIR):
        for map_dir in data_dir.iterdir():
            print(f"Processing map directory: {map_dir}")

            loaded_map = self.loader.load_map(map_dir=map_dir)
            if loaded_map is None:
                continue
            (info, diff_maps_tuples) = loaded_map

            for _diff_map, diff in diff_maps_tuples:
                stats = self.analyze_difficulty(diff, info)
                print(
                    f"[{info.songName}] NPS: {stats[1]:.2f} | Duration: {stats[0]:.0f}s | Layer Counts: {stats[2]}"
                )


if __name__ == "__main__":
    eda = BeatSaverEDA()
    eda.run()
