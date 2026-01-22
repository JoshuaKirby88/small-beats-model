from pathlib import Path

from small_beats_model.data_scraper import SCRAPED_DATA_DIR
from small_beats_model.generate import BeatGenerator
from small_beats_model.loader import MapLoader


class BeatGeneratorRunner:
    def __init__(self):
        self.beat_generator = BeatGenerator()
        self.loader = MapLoader()

    def run(self, audio_path: Path, output_file_name: str):
        predictions = self.beat_generator.infer(audio_path)
        output_path = self.beat_generator.decode_to_json(
            output_file_name=output_file_name, predictions=predictions
        )
        print(f"Saved to {output_path}")

    def run_many(self):
        for info_file, _, map_id in self.loader.iter_scraped():
            self.run(SCRAPED_DATA_DIR / map_id / info_file.songFilename, map_id)


if __name__ == "__main__":
    runner = BeatGeneratorRunner()
    runner.run_many()
