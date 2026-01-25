import ast
import json
from itertools import combinations, groupby

from small_beats_model.loader import VOCAB_PATH, MapLoader
from small_beats_model.models import DiffNote

EMPTY_TOKEN = 0
COMMON_GRID_STATE_COUNT = 100

NoteHash = tuple[int, int, int, int]
VocabHash = tuple[NoteHash, ...]
VocabType = dict[VocabHash, int]


class Vocab:
    def __init__(self):
        self.empty_grid_state: VocabHash = tuple([])

        if VOCAB_PATH.exists():
            with open(VOCAB_PATH, "r") as f:
                raw_vocab: dict[str, int] = json.load(f)
                self.vocab: VocabType = {
                    ast.literal_eval(k): v for k, v in raw_vocab.items()
                }
        else:
            self.vocab = {self.empty_grid_state: EMPTY_TOKEN}

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def group_notes_by_time(self, notes: list[DiffNote]):
        sorted_notes = sorted(notes, key=lambda x: x.time)
        return groupby(sorted_notes, key=lambda x: x.time)

    def hash_note(self, note: DiffNote):
        return (note.type, note.cutDirection, note.lineIndex, note.lineLayer)

    def unhash_note(self, time: float, hash: NoteHash):
        type, cutDirection, lineIndex, lineLayer = hash
        return DiffNote(
            _time=time,
            _type=type,
            _cutDirection=cutDirection,
            _lineIndex=lineIndex,
            _lineLayer=lineLayer,
        )

    def hash_grid_state(self, current_notes: list[DiffNote]):
        hashed_notes = [self.hash_note(n) for n in current_notes]
        sorted_hashed_notes = sorted(hashed_notes)
        return tuple(sorted_hashed_notes)

    def unhash_grid_state(self, time: float, hash: VocabHash):
        hashes = list(hash)
        notes = [self.unhash_note(time, hash) for hash in hashes]
        return notes

    def reduce_grid_to_subset(self, current_notes: list[DiffNote]) -> list[DiffNote]:
        for reduced_size in range(len(current_notes) - 1, 0, -1):
            for subset_tuple in combinations(current_notes, reduced_size):
                subset_notes = list(subset_tuple)
                hash = self.hash_grid_state(list(subset_notes))
                if hash in self.vocab:
                    return subset_notes
        return []

    def encode(self, current_notes: list[DiffNote]) -> int:
        hash = self.hash_grid_state(current_notes)
        if hash in self.vocab:
            return self.vocab[hash]
        else:
            subset_notes = self.reduce_grid_to_subset(current_notes)
            return self.encode(subset_notes)

    def decode(self, time, token: int):
        hash = self.inverse_vocab[token]
        notes = self.unhash_grid_state(time, hash)
        return notes


class VocabBuilder:
    def __init__(self):
        self.loader = MapLoader()
        self.vocab = Vocab()
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    def build(self):
        hash_counts: VocabType = {}

        for _, diff_tuples, _ in self.loader.iter_scraped():
            for _, diff_file in diff_tuples:
                for _, current_notes in self.vocab.group_notes_by_time(diff_file.notes):
                    hash = self.vocab.hash_grid_state(list(current_notes))
                    hash_counts[hash] = hash_counts.get(hash, 0) + 1

        common_hash_counts = {
            k: v for k, v in hash_counts.items() if v > COMMON_GRID_STATE_COUNT
        }

        sorted_common_hashes = sorted(
            common_hash_counts, key=lambda x: common_hash_counts[x]
        )

        vocab = {
            self.vocab.empty_grid_state: EMPTY_TOKEN,
            **{k: i + 1 for i, k in enumerate(sorted_common_hashes)},
        }

        serialized_vocab = {str(k): v for k, v in vocab.items()}

        with open(VOCAB_PATH, "w") as f:
            f.write(json.dumps(serialized_vocab, indent=2))

        print(f"Vocab size: {len(vocab)}")


if __name__ == "__main__":
    builder = VocabBuilder()
    builder.build()
