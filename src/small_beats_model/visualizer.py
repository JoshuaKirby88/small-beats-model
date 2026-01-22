import os
import subprocess
import time
from math import ceil
from pathlib import Path

from colorama import Fore, Style, init

from small_beats_model.loader import MapLoader
from small_beats_model.models import DiffNote
from small_beats_model.preprocessing import NUM_COLS, NUM_ROWS

VISUALIZER_STEP_S = 0.5


class Visualizer:
    def __init__(self):
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

    def visualize(self, diff_path: Path):
        diff_file = self.loader.load_prediction(diff_path)

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


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.play_audio(Path("data/raw/271dc/butterfly.egg"))
    visualizer.visualize(Path("data/predictions/271dc.dat"))
