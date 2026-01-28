import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

from small_beats_model.loader import SCRAPED_DATA_DIR
from small_beats_model.models import (
    BeatSaverDoc,
    BeatSaverResponse,
    DiffFile,
    InfoFile,
)
from small_beats_model.preprocessing import NUM_COLORS, NUM_COLS, NUM_ROWS

# Scrapes BeatSaver and saves the filtered result
# Info may contain reference to filtered diff files that do not exist


TEMP_DATA_DIR = Path("data/temp")

BEATSAVER_API_URL = "https://api.beatsaver.com"
SORT = "Rating"
PAGE_SIZE = 20
TOTAL_MAPS = 10_000
REQUEST_SLEEP_S = 0.5

CHARACTERISTIC_FILTERS = ["Standard"]


class BeatSaverScraper:
    def __init__(self):
        self.output_dir = SCRAPED_DATA_DIR
        self.temp_data_dir = TEMP_DATA_DIR
        self.api_url = BEATSAVER_API_URL
        self.sort = SORT
        self.page_size = PAGE_SIZE
        self.total_maps = TOTAL_MAPS
        self.characteristic_filters = CHARACTERISTIC_FILTERS
        self.request_sleep_s = REQUEST_SLEEP_S
        self.num_cols = NUM_COLS
        self.num_rows = NUM_ROWS
        self.num_colors = NUM_COLORS

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, page: int):
        url = f"{self.api_url}/search/text/{page}"
        headers = {"User-Agent": "SmallBeatsModelScraper"}
        params = {"order": self.sort, "pageSize": self.page_size}

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return BeatSaverResponse.model_validate(data)

    def download_map(self, doc: BeatSaverDoc):
        zip_path = (self.temp_data_dir / doc.id).with_suffix(".zip")
        unzip_dir = self.output_dir / doc.id

        response = requests.get(doc.versions[-1].downloadURL, stream=True)
        response.raise_for_status()

        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        zip_path.unlink()

        return unzip_dir

    def filter_map(self, map_dir: Path):
        info_path = map_dir / "Info.dat"
        with open(info_path, "r", encoding="utf-8") as f:
            raw_info_file = json.load(f)

        is_info_v2 = raw_info_file.get("_version", "").startswith("2.")

        if is_info_v2:
            info_file = InfoFile.model_validate(raw_info_file)
            diff_files = self.filter_diffs(map_dir, info_file)

            if len(diff_files) == 0:
                shutil.rmtree(map_dir)
        else:
            shutil.rmtree(map_dir)

    def filter_diffs(self, map_dir: Path, info_file: InfoFile):
        diff_files: list[DiffFile] = []

        for diff_set in info_file.difficultyBeatmapSets:
            for diff_map in diff_set.difficultyBeatmaps:
                diff_path = map_dir / diff_map.beatmapFilename
                with open(diff_path, "r", encoding="utf-8") as f:
                    raw_diff_file = json.load(f)
                is_diff_valid = (
                    diff_set.beatmapCharacteristicName in self.characteristic_filters
                    and raw_diff_file.get("_version", "").startswith("2.")
                    and len(raw_diff_file.get("_notes", [])) > 0
                )
                if is_diff_valid:
                    diff_file = DiffFile.model_validate(raw_diff_file)
                    is_notes_valid = all(
                        note.lineLayer in range(self.num_rows)
                        and note.lineIndex in range(self.num_cols)
                        and note.type in range(self.num_colors)
                        for note in diff_file.notes
                    )
                    if is_notes_valid:
                        diff_files.append(diff_file)
                    else:
                        os.remove(diff_path)
                else:
                    os.remove(diff_path)

        return diff_files

    def run(self):
        docs: list[BeatSaverDoc] = []

        for i in range(0, self.total_maps, self.page_size):
            if i > 0:
                time.sleep(self.request_sleep_s)
            page = i // self.page_size
            response = self.fetch(page=page)
            docs.extend(response.docs)
            print(
                f"Fetch: [{i} / {self.total_maps}] {round(i * 100 / self.total_maps, 2)}%"
            )

        for i, doc in enumerate(docs):
            try:
                unzip_dir = self.download_map(doc)
                self.filter_map(unzip_dir)
                print(f"Download: [{i} / {len(docs)}] {round(i * 100 / len(docs), 2)}%")
            except Exception as e:
                print(f"Error processing map {doc.id}: {e}")


if __name__ == "__main__":
    scraper = BeatSaverScraper()
    scraper.run()
