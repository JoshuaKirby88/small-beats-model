from small_beats_model.loader import MapLoader
from small_beats_model.models import DiffFile, InfoFile
from small_beats_model.preprocessing import NUM_COLS


class BeatSaverEDA:
    def __init__(self):
        self.loader = MapLoader()

    def analyze_difficulty(self, info_file: InfoFile, diff_file: DiffFile):
        last_beat_time = diff_file.notes[-1].time
        bpm = info_file.beatsPerMinute
        last_time_s = (last_beat_time / bpm) * 60

        total_notes = len(diff_file.notes)
        nps = total_notes / last_time_s

        layer_counts = {i: i for i in range(NUM_COLS)}
        for n in diff_file.notes:
            layer_counts[n.lineLayer] += 1

        return last_time_s, nps, layer_counts

    def run(self):
        for info_file, diff_tuples, map_id in self.loader.iter_scraped():
            print(f"Processing: {map_id}")

            for _, diff_file in diff_tuples:
                stats = self.analyze_difficulty(info_file, diff_file)
                print(
                    f"[{info_file.songName}] NPS: {stats[1]:.2f} | Duration: {stats[0]:.0f}s | Layer Counts: {stats[2]}"
                )


if __name__ == "__main__":
    eda = BeatSaverEDA()
    eda.run()
