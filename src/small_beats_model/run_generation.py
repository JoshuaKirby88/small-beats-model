from datetime import datetime
from pathlib import Path

import typer

from small_beats_model.data_scraper import SCRAPED_DATA_DIR
from small_beats_model.generate import BeatGenerator
from small_beats_model.loader import MapLoader


class BeatGeneratorRunner:
    def __init__(self):
        self.scraped_data_dir = SCRAPED_DATA_DIR
        self.beat_generator = BeatGenerator()
        self.loader = MapLoader()

    def run(self, audio_path: Path):
        print("Generating...")
        predictions = self.beat_generator.infer(audio_path)
        notes = self.beat_generator.tokens_to_notes(predictions)
        output_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = self.beat_generator.save(
            output_dir_name=output_dir_name, predictions=predictions, notes=notes
        )
        print(f"Saved to {output_dir}")


app = typer.Typer()


@app.command()
def main(audio_path: Path | None = None):
    audio_path = audio_path or Path(typer.prompt("Path to audio file"))

    runner = BeatGeneratorRunner()
    runner.run(audio_path=audio_path)


if __name__ == "__main__":
    app()
