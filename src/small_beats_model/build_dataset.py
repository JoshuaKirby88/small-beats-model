import json
from pathlib import Path

import torch

from small_beats_model.loader import MapLoader
from small_beats_model.models import DatasetMeta
from small_beats_model.preprocessing import AudioProcessor, LabelProcessor

DATA_DIR = Path("data/raw")
DATASET_DIR = Path("data/processed")


class DatasetBuilder:
    def __init__(self, data_dir=DATA_DIR, dataset_dir=DATASET_DIR):
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.loader = MapLoader()
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()

        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir(parents=True)

        self.label_processor.build_vocab()

    def build(self):
        for map_dir in self.data_dir.iterdir():
            loaded_map = self.loader.load_map(map_dir=map_dir)
            if loaded_map is None:
                continue
            (info, diff_map_tuples) = loaded_map

            audio_path = map_dir / info.songFilename
            audio_tensor = self.audio_processor.process_audio(audio_path)

            for diff_map, diff in diff_map_tuples:
                audio_duration_s = self.audio_processor.get_duration_s(audio_tensor)
                total_beats = (audio_duration_s / 60) * info.beatsPerMinute
                label_tensor = self.label_processor.notes_to_grid(
                    notes=diff.notes, total_beats=total_beats
                )

                feature_dir = self.dataset_dir / f"{map_dir.name}_{diff_map.difficulty}"
                feature_dir.mkdir(parents=True, exist_ok=True)

                meta = DatasetMeta(
                    bpm=info.beatsPerMinute,
                    njs=diff_map.noteJumpMovementSpeed,
                    njOffset=diff_map.noteJumpStartBeatOffset,
                    difficulty=diff_map.difficulty,
                    song_duration_s=audio_duration_s,
                    total_beats=len(diff.notes),
                )

                audio_tensor_file = feature_dir / "features.pt"
                torch.save(audio_tensor, audio_tensor_file)
                label_tensor_file = feature_dir / "labels.pt"
                torch.save(label_tensor, label_tensor_file)
                meta_file = feature_dir / "meta.json"
                with open(meta_file, "w") as f:
                    f.write(json.dumps(meta.model_dump()))


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
