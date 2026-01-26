import json
import random
import shutil
import string
import subprocess
import unicodedata
from pathlib import Path

from anyascii import anyascii

from small_beats_model.loader import (
    EXPORT_DIR,
    PLACEHOLDER_EXPORT_DIR,
    MapLoader,
)
from small_beats_model.models import InfoDiff, InfoDiffSet, InfoFile
from small_beats_model.preprocessing import AudioProcessor


class Packager:
    def __init__(self):
        self.loader = MapLoader()
        self.audio_processor = AudioProcessor()

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    def generate_id(self, length: int):
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choices(chars, k=length))

    def sanitize_path(self, path: str):
        normalized = unicodedata.normalize("NFC", path)
        ascii = anyascii(normalized)
        return ascii.strip()

    def generate_diff_set(self):
        return InfoDiffSet(
            _beatmapCharacteristicName="Standard",
            _difficultyBeatmaps=[
                InfoDiff(
                    _difficulty="Expert",
                    _difficultyRank=7,
                    _beatmapFilename="Expert.dat",
                    _noteJumpMovementSpeed=16,
                    _noteJumpStartBeatOffset=0,
                )
            ],
        )

    def generate_info(
        self,
        song_name: str,
        song_author_name: str,
        song_file_name: str,
        author_name: str,
        cover_file_name: str,
        bpm: float,
    ):
        return InfoFile(
            _version="2.0.0",
            _songName=song_name,
            _songSubName="",
            _songAuthorName=song_author_name,
            _levelAuthorName=author_name,
            _beatsPerMinute=bpm,
            _songTimeOffset=0,
            _shuffle=0,
            _shufflePeriod=0,
            _previewStartTime=0,
            _previewDuration=10,
            _songFilename=song_file_name,
            _coverImageFilename=cover_file_name,
            _environmentName="DefaultEnvironment",
            _difficultyBeatmapSets=[self.generate_diff_set()],
        )

    def convert_audio(self, audio_path: Path, egg_output_path: Path):
        if audio_path.suffix.lower() in [".egg", ".ogg"]:
            shutil.copy(audio_path, egg_output_path)
        else:
            ogg_output_path = egg_output_path.with_suffix(".ogg")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-map_metadata",
                    "-1",
                    "-c:a",
                    "libvorbis",
                    "-q:a",
                    "4",
                    str(ogg_output_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            shutil.move(ogg_output_path, egg_output_path)

    def copy_cover(self, cover_path: Path, output_dir: Path):
        if cover_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            return self.copy_cover(PLACEHOLDER_EXPORT_DIR / "cover.jpg", output_dir)
        else:
            output_path = output_dir / f"cover{cover_path.suffix.lower()}"
            shutil.copy(cover_path, output_path)
            return output_path

    def package(
        self,
        diff_path: Path,
        audio_path: Path,
        cover_path: Path,
        song_name: str,
        song_author_name: str,
        author_name: str,
    ):
        id = self.generate_id(5)
        map_export_dir = (
            EXPORT_DIR
            / f"{id} ({self.sanitize_path(song_name)} - {self.sanitize_path(author_name)})"
        )
        map_export_dir.mkdir(parents=True, exist_ok=True)

        audio_export_path = map_export_dir / "song.egg"
        self.convert_audio(audio_path=audio_path, egg_output_path=audio_export_path)

        cover_export_path = self.copy_cover(cover_path, map_export_dir)

        bpm = self.audio_processor.get_bpm(audio_export_path)
        info_file = self.generate_info(
            song_name=song_name,
            song_author_name=song_author_name,
            song_file_name=audio_export_path.name,
            author_name=author_name,
            cover_file_name=cover_export_path.name,
            bpm=bpm,
        )
        with open(map_export_dir / "Info.dat", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    info_file.model_dump(by_alias=True), indent=2, ensure_ascii=False
                )
            )

        diff_file = self.loader.load_diff_map(diff_path)
        with open(map_export_dir / "Expert.dat", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    diff_file.model_dump(by_alias=True), indent=2, ensure_ascii=False
                )
            )

        return map_export_dir


if __name__ == "__main__":
    packager = Packager()
    packager.package(
        diff_path=Path("data/predictions/Bのリベンジ/Expert.dat"),
        audio_path=Path("data/inputs/Bのリベンジ/song.m4a"),
        cover_path=Path("data/inputs/Bのリベンジ/cover.jpg"),
        song_name="Bのリベンジ",
        song_author_name="B小町",
        author_name="Joshua Kirby",
    )
