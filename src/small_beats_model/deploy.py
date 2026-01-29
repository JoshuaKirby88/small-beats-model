import subprocess
import sys
from pathlib import Path

import typer

QUEST_MOD_DIR = Path(
    "/sdcard/ModData/com.beatgames.beatsaber/Mods/SongCore/CustomLevels"
)
ADB = "adb"


class Deployer:
    def __init__(self):
        self.adb = ADB
        self.quest_mod_dir = QUEST_MOD_DIR

    def run_adb(self, args: list[str]):
        try:
            return subprocess.run(
                [self.adb] + args, capture_output=True, text=True, check=True
            )
        except FileNotFoundError:
            print("Error: `adb` not found. Install Android Platform Tools.")
            sys.exit(1)

    def get_device(self):
        results = self.run_adb(["devices"])

        lines = [line.strip() for line in results.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            print("Error: No devices found. Connect your Quest via USB.")
            return None

        devices = lines[1:]
        valid_devices: list[str] = []

        for line in devices:
            parts = line.split()
            if len(parts) < 2:
                continue
            device, status = parts[0], parts[1]

            if status == "unauthorized":
                print(
                    f"Error: Device {device} unauthorized. 'Allow' connection in headset."
                )
                return None
            elif status == "offline":
                print(f"Error: Device {device} offline.")
                return None
            elif status == "device":
                valid_devices.append(device)
        if len(valid_devices) < 1:
            print("Error: No valid devices.")
            return None

        if len(valid_devices) > 1:
            print(
                f"Error: Multiple devices found ({', '.join(valid_devices)}). Connect only one."
            )
            return None
        return valid_devices[0]

    def verify_model(self, device: str):
        results = self.run_adb(["-s", device, "shell", "getprop", "ro.product.model"])
        model = results.stdout.strip()

        if "quest" not in model.lower():
            print(f"Error: Device '{model}' is not a Quest.")
            return None

        print(f"Connected to '{model}'")
        return model

    def verify(self):
        device = self.get_device()
        if device is None:
            sys.exit(1)

        model = self.verify_model(device)
        if model is None:
            sys.exit(1)

        return device, model

    def push_to_quest(self, exported_map_dir: Path):
        device, _ = self.verify()

        quest_map_path = self.quest_mod_dir / exported_map_dir.name
        print(f"Pushing '{exported_map_dir.name}' to Quest...")

        results = self.run_adb(
            [
                "-s",
                device,
                "push",
                str(exported_map_dir),
                str(self.quest_mod_dir.as_posix()),
            ]
        )

        if results.returncode == 0:
            print(f"Success! Map pushed to: {quest_map_path}")
        else:
            print("Error pushing files:")
            print(results.stderr)


app = typer.Typer()


@app.command()
def main(map_dir: Path | None):
    map_dir = map_dir or Path(typer.prompt("Path to map folder"))

    deployer = Deployer()
    deployer.push_to_quest(exported_map_dir=map_dir)


if __name__ == "__main__":
    app()
