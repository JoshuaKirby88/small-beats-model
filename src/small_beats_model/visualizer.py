import os
import subprocess
import time
from math import ceil
from pathlib import Path

from colorama import Fore, Style, init

from small_beats_model.loader import PREDICTION_DIR, SCRAPED_DATA_DIR, MapLoader
from small_beats_model.models import DiffNote
from small_beats_model.preprocessing import NUM_COLS, NUM_ROWS

VISUALIZER_STEP_S = 0.25


class Visualizer:
    def __init__(self):
        self.scraped_data_dir = SCRAPED_DATA_DIR
        self.prediction_dir = PREDICTION_DIR
        self.visualizer_step_s = VISUALIZER_STEP_S
        self.arrows = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘", "•"]
        self.audio_process: subprocess.Popen[bytes] | None = None
        self.loader = MapLoader()

        init()

    def play_audio(self, audio_path: Path):
        self.audio_process = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(audio_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def render_frame(self, notes_at_time: list[DiffNote]):
        os.system("clear")
        grid = [[" " for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]

        for note in notes_at_time:
            color = Fore.RED if note.type == 0 else Fore.BLUE
            arrow = self.arrows[note.cutDirection]
            grid[note.lineLayer][note.lineIndex] = f"{color}{arrow}{Style.RESET_ALL}"

        for row in reversed(grid):
            print(" ".join(f"[{cell}]" for cell in row))

    def visualize(self, prediction_diff_dir: Path):
        diff_file = self.loader.load_prediction_diff_map(prediction_diff_dir)

        max_time = ceil(diff_file.notes[-1].time)
        time_steps = [
            i * self.visualizer_step_s
            for i in range(max_time * ceil(1 / self.visualizer_step_s))
        ]

        for t in time_steps:
            notes_at_time = [n for n in diff_file.notes if n.time == t]
            self.render_frame(notes_at_time)
            time.sleep(self.visualizer_step_s)

        if self.audio_process is not None:
            self.audio_process.terminate()

    def visualize_map_id(self, map_id: str):
        audio_path = self.loader.get_audio_path(map_id)
        prediction_diff_dir = self.prediction_dir / map_id
        self.play_audio(audio_path)
        self.visualize(prediction_diff_dir)


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.visualize_map_id("1a0b6")
