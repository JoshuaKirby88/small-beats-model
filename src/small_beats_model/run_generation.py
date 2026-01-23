from pathlib import Path

from small_beats_model.data_scraper import SCRAPED_DATA_DIR
from small_beats_model.generate import BeatGenerator
from small_beats_model.loader import MapLoader


class BeatGeneratorRunner:
    def __init__(self):
        self.scraped_data_dir = SCRAPED_DATA_DIR
        self.beat_generator = BeatGenerator()
        self.loader = MapLoader()

    def run(self, audio_path: Path, output_dir_name: str):
        predictions = self.beat_generator.infer(audio_path)
        output_dir = self.beat_generator.save(
            output_dir_name=output_dir_name, predictions=predictions
        )
        print(f"Saved to {output_dir}")

    def run_on_scraped(self, max: int):
        for i, (info_file, _, map_id) in enumerate(self.loader.iter_scraped()):
            if i >= max:
                break
            audio_path = self.scraped_data_dir / map_id / info_file.songFilename
            self.run(audio_path=audio_path, output_dir_name=map_id)


if __name__ == "__main__":
    runner = BeatGeneratorRunner()
    runner.run(
        audio_path=Path("data/inputs/Bのリベンジ/song.m4a"),
        output_dir_name="Bのリベンジ",
    )
