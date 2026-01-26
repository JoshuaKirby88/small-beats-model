import os
import subprocess
import time
from itertools import groupby
from pathlib import Path

from colorama import Fore, Style, init

from small_beats_model.loader import PREDICTION_DIR, SCRAPED_DATA_DIR, MapLoader
from small_beats_model.models import DiffNote
from small_beats_model.preprocessing import (
    NUM_COLS,
    NUM_ROWS,
    STEPS_PER_BEAT,
    AudioProcessor,
)

VISUALIZER_STEP_S = 0.25


class Visualizer:
    def __init__(self):
        self.scraped_data_dir = SCRAPED_DATA_DIR
        self.prediction_dir = PREDICTION_DIR
        self.visualizer_step_s = VISUALIZER_STEP_S
        self.steps_per_beat = STEPS_PER_BEAT
        self.arrows = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘", "•"]
        self.audio_process: subprocess.Popen[bytes] | None = None
        self.loader = MapLoader()
        self.audio_processor = AudioProcessor()

        init()

    def play_audio(self, audio_path: Path):
        self.audio_process = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(audio_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def clear(self):
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

    def render_frame(self, current_notes: list[DiffNote]):
        self.clear()
        grid = [[" " for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]

        for note in current_notes:
            color = Fore.RED if note.type == 0 else Fore.BLUE
            arrow = self.arrows[note.cutDirection]
            grid[note.lineLayer][note.lineIndex] = f"{color}{arrow}{Style.RESET_ALL}"

        for row in reversed(grid):
            print(" ".join(f"[{cell}]" for cell in row))

    def render_diff_map(self, diff_path: Path, bpm: float):
        diff_file = self.loader.load_diff_map(diff_path)
        current_beat = 0
        for note_time, notes in groupby(diff_file.notes, key=lambda x: x.time):
            delay_beats = note_time - current_beat
            delay_s = delay_beats * 60 / bpm
            time.sleep(delay_s)
            self.render_frame(list(notes))
            current_beat = note_time

        if self.audio_process is not None:
            self.audio_process.terminate()

    def visualize(self, audio_path: Path, diff_path: Path):
        self.play_audio(audio_path)
        bpm = self.audio_processor.get_bpm(audio_path)
        print(bpm)
        self.render_diff_map(diff_path, bpm)


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.visualize(
        audio_path=Path("data/exports/89qcn (b-no-revenge - Joshua Kirby)/song.egg"),
        diff_path=Path("data/predictions/Bのリベンジ/Expert.dat"),
    )
