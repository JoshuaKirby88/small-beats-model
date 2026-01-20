import json
import time
import zipfile
from pathlib import Path

import requests

from small_beats_model.models import BeatSaverMap, BeatSaverResponse

# Figure out disk size of 1 sample = ~20MB

BEATSAVER_API_URL = "https://api.beatsaver.com"
DATA_DIR = Path("data/raw")
TEMP_DATA_DIR = Path("data/temp")
SORT = "Rating"
PAGE_SIZE = 20
TOTAL_MAPS = 100
REQUEST_SLEEP_S = 0.5


class BeatSaverScraper:
    def __init__(self, output_dir: Path = DATA_DIR, temp_data_dir=TEMP_DATA_DIR):
        self.output_dir = output_dir
        self.temp_data_dir = temp_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_latest_maps(
        self, api_url=BEATSAVER_API_URL, sort=SORT, page=0, page_size=PAGE_SIZE
    ) -> list[BeatSaverMap]:
        url = f"{api_url}/search/text/{page}"
        headers = {"User-Agent": "SmallBeatsModelScraper"}
        params = {"order": sort, "pageSize": page_size}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(json.dumps(data))
            return BeatSaverResponse.model_validate(data).docs
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {e}")
            return []

    def download_map(self, beat_saver_map: BeatSaverMap) -> bool:
        zip_path = self.temp_data_dir / f"{beat_saver_map.id}.zip"
        unzip_dir = self.output_dir / f"{beat_saver_map.id}"

        response = requests.get(beat_saver_map.versions[-1].downloadURL, stream=True)
        response.raise_for_status()

        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)

        zip_path.unlink()

        return True

    def run(self, total_maps=TOTAL_MAPS, page_size=PAGE_SIZE) -> None:
        all_maps: list[BeatSaverMap] = []

        for start in range(0, total_maps, page_size):
            time.sleep(REQUEST_SLEEP_S)
            end = min(start + page_size, total_maps)
            page = start // page_size
            maps = self.fetch_latest_maps(page=page, page_size=end - start)
            all_maps.extend(maps)

        for row in all_maps:
            time.sleep(REQUEST_SLEEP_S)
            self.download_map(row)


if __name__ == "__main__":
    scraper = BeatSaverScraper()
    scraper.run(total_maps=10, page_size=10)
