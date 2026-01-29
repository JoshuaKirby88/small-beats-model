# Small Beats Model 🎵⚔️🟥🟦

A small-scale ML model that generates Beat Saber maps from music.

## What it does

Takes an audio file, spits out a Beat Saber map. V0 is functional but rough around the edges.

## Why

Learning ML. Also wanted more niche songs to play in Beat Saber.

## Setup

### 1. Prerequisites

Requirements:
*   **uv**: Required to run everything.
*   **ffmpeg**: Required for audio processing.
*   **adb**: Required only if you want to deploy maps directly to a Meta Quest.

### 2. Hardware Support

Supports NVIDIA GPUs (CUDA), Apple Silicon (MPS), and CPU as a fallback.
The model can generate a map for any audio format.
The script to transfer the generated map to your VR only works for Quest.

You can visualize generated maps by zipping them and uploading it to [ArcViewer](https://allpoland.github.io/ArcViewer).

## Usage

### **Generate**, **Package**, and **Deploy** to `Quest`:


This will prompt you for each parameter:
```
uv run src/orchestrator.py
```

Or provide all arguments:
```
uv run src/orchestrator.py \
	--audio-path=song.mp3 \  # Any audio file
	--cover-path=cover.jpg \  # The cover image that will appear in Beat Saber (jpg|jpeg|png) (optional)
	--song-name="Crab Rave" \  # Name of the song
	--song-author-name=Noisestorm \  # Song author
	--author-name=Josh  # Map creator (your) name
```

The above commands will:
1. **Generate** a map (`Expert.dat`) and also dump the raw tokens (`tokens.json`) in `data/predictions/{timestamp}/`
2. **Package** the map in `data/exports/{id} ({song_name} - {author_name})`
3. **Deploy** to your Quest using `adb push 'data/exports/{id} ({song_name} - {author_name})' '/sdcard/ModData/com.beatgames.beatsaber/Mods/SongCore/CustomLevels'`


### Just **Generate**:

```
uv run src/small_beats_model/run_generation.py \
	--audio-path=song.mp3  # Any audio file
```

### Just **Package**:

```
uv run src/small_beats_model/package.py \
	--diff-path=data/predictions/2026-01-28_15-00-40/Expert.dat \
	--audio-path=song.mp3 \
	--cover-path=cover.jpg \
	--song-name="Crab Rave" \
	--song-author-name=Noisestorm \
	--author-name=Josh
```

### Just **Deploy** to `Quest`:

Make sure your Quest is connected via USB and in developer mode.

```
uv run src/small_beats_model/deploy.py \
	--map-dir=data/exports/0ipu0 (Crab Rave - Josh) # Must contain Info.dat, Expert.dat, a .egg audio, and a cover image
```

## Status

V0 works. Feel free to open issues if you try it out.

# Details

## How it Works

**Vocabulary:**

The model has a vocab size of **1,005**.
I tokenized all unique grid states in the training data, and only kept the grid states that appeared more than **50** times (~0.00001%) in the dataset.
This reduced the vocab size from **5,955** to **1,005**.
I handled the _discarded_ grid states by performing a subset reduction operation (remove random blocks until it matches one of the tokens in the vocab).

**Model:**

- **Architecture:** A hybrid CNN + GRU network.
	A 1D Convolutional encoder extracts features from the audio (MFCCs), which are fed into a 3-layer GRU.
- **Autoregressive:** The model predicts the next grid state based on the current audio window and the previously generated token.
- **Attention:** A self-attention mechanism is applied over the RNN outputs to refine the context before the final classification head.
- **BPM Normalization:** All input audio is stretched/squashed to 120 BPM.
	This allows the model to focus on the audio and the notes without having to learn the tempo.
	Conveniently, this allows for the tempo of the song to naturally control the difficulty of the generated map.
- **Class Weighting:** Uses inverse frequency weighting to incentivize the model to generate notes, counteracting the fact that ~70% of the training data is empty grids.

**Inference:**

- Generates the map beat-by-beat using a sliding window context.
- Uses **Top-K sampling** with **temperature** to ensure variety while keeping the map coherent.
- Post-processing converts the generated grid tokens back into standard Beat Saber JSON format (`Expert.dat`).

## Training Data

All data was collected from the [BeatSaver API](https://api.beatsaver.com/docs/index.html).
The model was trained on the top **~9,000** V2 maps sorted by `Rating`.
Huge thanks to the mapping community. I've spent countless hours playing your maps, and this project wouldn't have been possible without you.
