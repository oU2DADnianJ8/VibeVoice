# VibeVoice Text-to-Speech API

This directory contains a FastAPI-based server that exposes an OpenAI-compatible `/v1/audio/speech` endpoint powered by the VibeVoice model.

## Prerequisites

1. **Python** 3.9 or newer is recommended.
2. **FFmpeg** is required for MP3 encoding. Install it via your package manager (e.g. `sudo apt-get install ffmpeg`) or download it from [ffmpeg.org](https://ffmpeg.org/download.html).

## Installation

1. Create and activate a Python virtual environment (optional but recommended).
2. Install the dependencies:

```bash
pip install -r api/requirements.txt
```

The requirements file installs the VibeVoice package from the repository root in editable mode (`-e ..`). Run the command from the project root so that the relative path resolves correctly.

## Voice Presets

Place reference voice recordings in `api/voices`. The server automatically scans this directory on startup. Each file name (without extension) becomes the `voice` identifier in API requests.

Supported file types include `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, and `.aac`. Audio is internally resampled to 24 kHz.

## Running the Server

You can launch the server with either of the following approaches:

### 1. Python entry point

```bash
python api/server.py \
    --model_path microsoft/VibeVoice-1.5b \
    --device auto \
    --host 0.0.0.0 \
    --port 8000
```

### 2. Uvicorn CLI

```bash
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5b \
VIBEVOICE_DEVICE=auto \
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Configuration Options

| Option | CLI Flag | Environment Variable | Default |
|--------|----------|----------------------|---------|
| Model path | `--model_path` | `VIBEVOICE_MODEL_PATH` | `microsoft/VibeVoice-1.5b` |
| Compute device (`cuda`, `mps`, `cpu`, or `auto`) | `--device` | `VIBEVOICE_DEVICE` | Auto-detected |
| Host | `--host` | `VIBEVOICE_HOST` | `0.0.0.0` |
| Port | `--port` | `VIBEVOICE_PORT` | `8000` |
| CFG scale | `--cfg_scale` | `VIBEVOICE_CFG_SCALE` | `1.3` |
| Diffusion inference steps | `--inference_steps` | `VIBEVOICE_INFERENCE_STEPS` | `5` |
| Voices directory | `--voices_dir` | `VIBEVOICE_VOICES_DIR` | `api/voices` |
| Output directory for saved MP3 files | `--output_dir` | `VIBEVOICE_OUTPUT_DIR` | `api/output` |

## API Usage

Send a POST request to `/v1/audio/speech` with a JSON body mirroring the OpenAI Text-to-Speech API. Example:

```json
{
  "model": "microsoft/VibeVoice-1.5b",
  "input": "Hello from VibeVoice!",
  "voice": "en-Alice_woman",
  "response_format": "mp3",
  "cfg_scale": 1.3,
  "stream": true
}
```

Responses stream audio by default so playback can begin as soon as data is available. Set `"stream": false` to receive the complete file in a single response.

Supported `response_format` values are `mp3`, `wav`, and `flac`. Regardless of the requested format, an MP3 copy of the generated audio is saved to the configured output directory (default `api/output`).

Errors are returned in JSON with appropriate HTTP status codes if the input is invalid, a voice preset is missing, or speech synthesis fails.
