from small_beats_model.loader import MapLoader
from small_beats_model.models import DiffFile, InfoFile
from small_beats_model.preprocessing import NUM_COLORS, NUM_COLS


class BeatSaverEDA:
    def __init__(self):
        self.num_colors = NUM_COLORS
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

    def analyze_notes(self, diff_file: DiffFile):
        violating_notes = 0
        times = set(n.time for n in diff_file.notes)
        for time in times:
            notes = [n for n in diff_file.notes if n.time == time]
            zero_notes = [n for n in notes if n.type == 0]
            one_notes = [n for n in notes if n.type == 1]
            if len(zero_notes) > 1 or len(one_notes) > 1:
                violating_notes += 1
        return violating_notes, len(diff_file.notes)

    def run(self):
        total_violating_notes = 0
        all_total_notes = 0
        total_diffs = 0

        for info_file, diff_tuples, map_id in self.loader.iter_scraped():
            print(f"Processing: {map_id}")

            for _, diff_file in diff_tuples:
                total_diffs += 1
                violating_notes, total_notes = self.analyze_notes(diff_file)
                total_violating_notes += violating_notes
                all_total_notes += total_notes

                stats = self.analyze_difficulty(info_file, diff_file)
                print(
                    f"[{info_file.songName}] NPS: {stats[1]:.2f} | Duration: {stats[0]:.0f}s | Layer Counts: {stats[2]}"
                )

        print(f"Total diffs: {total_diffs}")
        print(
            f"Total violating notes: [{total_violating_notes} / {all_total_notes}] {round((total_violating_notes / all_total_notes) * 100, 2)}%)"
        )


if __name__ == "__main__":
    eda = BeatSaverEDA()
    eda.run()
