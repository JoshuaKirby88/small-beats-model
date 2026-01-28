from itertools import groupby

from small_beats_model.loader import MapLoader
from small_beats_model.models import DiffNote
from small_beats_model.preprocessing import NUM_COLORS


class BeatSaverEDA:
    def __init__(self):
        self.num_colors = NUM_COLORS
        self.loader = MapLoader()

    def calculate_unique_steps(self, notes: list[DiffNote]):
        counts = {}
        grouped_notes = groupby(notes, key=lambda x: x.time)
        for _, current_notes in grouped_notes:
            hashable_notes = [
                (n.type, n.cutDirection, n.lineLayer, n.lineIndex)
                for n in current_notes
            ]
            sorted_notes = sorted(hashable_notes)
            notes_hash = tuple(sorted_notes)
            counts[notes_hash] = counts.get(notes_hash, 0) + 1

        print("Unique combination of notes at each step (appearing more than x)")
        for i in range(0, 1001, 50):
            more_than_i_counts = {k: v for k, v in counts.items() if v > i}
            print(
                f"{i} ({i / len(notes) * 100:.4f}% of total): {len(more_than_i_counts)}"
            )

    def run(self):
        maps = list(self.loader.iter_scraped())
        diff_files = [diff_tuple[1] for map in maps for diff_tuple in map[1]]
        notes = [n for d in diff_files for n in d.notes]

        print("Total diffs:", len(diff_files))
        print("Total notes:", len(notes))

        self.calculate_unique_steps(notes)


if __name__ == "__main__":
    eda = BeatSaverEDA()
    eda.run()
