from pathlib import Path

from small_beats_model.generate import BeatGenerator


class BeatGeneratorRunner:
    def __init__(self):
        self.beat_generator = BeatGenerator()

    def run(self, audio_path: Path):
        predictions = self.beat_generator.infer(audio_path)
        output_path = self.beat_generator.decode_to_json(audio_path, predictions)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    runner = BeatGeneratorRunner()
    runner.run(
        Path("/Users/joshua/Developer/small-beats-model/data/raw/1a0b6/empires1.egg")
    )
