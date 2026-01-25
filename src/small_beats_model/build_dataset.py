import json

import torch

from small_beats_model.data_scraper import SCRAPED_DATA_DIR
from small_beats_model.loader import DATASET_DIR, MapLoader
from small_beats_model.models import DatasetMeta
from small_beats_model.preprocessing import AudioProcessor, LabelProcessor


class DatasetBuilder:
    def __init__(self):
        self.loader = MapLoader()
        self.audio_processor = AudioProcessor()
        self.label_processor = LabelProcessor()

        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    def build(self):
        for i, (info_file, diff_tuples, map_id) in enumerate(
            self.loader.iter_scraped()
        ):
            audio_path = SCRAPED_DATA_DIR / map_id / info_file.songFilename
            audio_tensor = self.audio_processor.process_audio(
                song_offset=info_file.songTimeOffset, audio_path=audio_path
            )

            for diff_map, diff_file in diff_tuples:
                feature_dir = DATASET_DIR / f"{map_id}_{diff_map.difficulty}"
                feature_dir.mkdir(parents=True, exist_ok=True)

                audio_duration_s = self.audio_processor.get_duration_s(audio_tensor)
                torch.save(audio_tensor, feature_dir / "features.pt")

                total_beats = (audio_duration_s / 60) * info_file.beatsPerMinute
                label_tensor = self.label_processor.notes_to_grid(
                    notes=diff_file.notes, total_beats=total_beats
                )
                torch.save(label_tensor, feature_dir / "labels.pt")

                meta = DatasetMeta(
                    bpm=info_file.beatsPerMinute,
                    njs=diff_map.noteJumpMovementSpeed,
                    njOffset=diff_map.noteJumpStartBeatOffset,
                    difficulty=diff_map.difficulty,
                    song_duration_s=audio_duration_s,
                    total_beats=total_beats,
                )
                with open(feature_dir / "meta.json", "w") as f:
                    f.write(json.dumps(meta.model_dump()))

            if i % 10 == 0:
                print(f"Progress: {i + 1}")


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
