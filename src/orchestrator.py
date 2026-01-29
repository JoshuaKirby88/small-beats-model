from pathlib import Path
from typing import cast

import typer

from small_beats_model.deploy import Deployer
from small_beats_model.generate import BeatGenerator
from small_beats_model.package import Packager


class Orchestrator:
    def __init__(self):
        self.generator = BeatGenerator()
        self.packager = Packager()
        self.deployer = Deployer()

    def main(
        self,
        audio_path: Path,
        cover_path: Path | None,
        song_name: str,
        song_author_name: str,
        author_name: str,
    ):
        self.deployer.verify()

        print("Generating...")
        predictions = self.generator.infer(audio_path)
        notes = self.generator.tokens_to_notes(predictions)

        print("Saving...")
        output_dir = self.generator.save(
            output_dir_name=song_name, predictions=predictions, notes=notes
        )

        exported_map_dir = self.packager.package(
            diff_path=output_dir / "Expert.dat",
            audio_path=audio_path,
            cover_path=cover_path,
            song_name=song_name,
            song_author_name=song_author_name,
            author_name=author_name,
        )

        self.deployer.push_to_quest(exported_map_dir=exported_map_dir)


app = typer.Typer()


@app.command()
def generate(
    audio_path: Path | None = None,
    cover_path: Path | None = None,
    song_name: str | None = None,
    song_author_name: str | None = None,
    author_name: str | None = None,
):
    is_interactive = not any(
        [
            audio_path,
            cover_path,
            song_name,
            song_author_name,
            author_name,
        ]
    )

    audio_path = audio_path or Path(typer.prompt("Path to audio file"))
    if cover_path is None and is_interactive:
        val = typer.prompt(
            "Path to cover image (jpg/png) [Enter to skip]",
            default="",
            show_default=False,
        )
        cover_path = Path(val) if val else None
    song_name = song_name or cast(str, typer.prompt("Song name"))
    song_author_name = song_author_name or cast(
        str, typer.prompt("Original song author")
    )
    author_name = author_name or cast(str, typer.prompt("Map creator name"))

    orchestrator = Orchestrator()
    orchestrator.main(
        audio_path=audio_path,
        cover_path=cover_path,
        song_name=song_name,
        song_author_name=song_author_name,
        author_name=author_name,
    )


if __name__ == "__main__":
    app()
