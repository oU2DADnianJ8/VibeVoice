# VibeVoice OpenAI-Compatible TTS Server

This directory contains a production-ready FastAPI server that exposes an OpenAI-compatible `/v1/audio/speech` endpoint backed by the **VibeVoice** text-to-speech model. The service mirrors OpenAI's request schema while enabling the use of custom voice presets stored locally.

## Features

- ✅ FastAPI application with OpenAI-compatible request/response semantics.
- ✅ Automatic model loading with device auto-detection (`cuda`, `mps`, or `cpu`).
- ✅ Voice preset discovery from the local `voices/` directory, with caching and resampling.
- ✅ Support for MP3, WAV, and FLAC outputs using in-memory streaming.
- ✅ Additional helper endpoints: `/health` for readiness checks and `/voices` to list discovered presets.

## Prerequisites

1. **Python 3.9+**
2. **FFmpeg** in your `PATH` if you plan to export MP3 audio (required by `pydub`).
3. Access to a local copy of the VibeVoice model weights.
4. Optional but recommended: a Python virtual environment.

## Installation

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r api/requirements.txt
   ```

   The final line installs the local `vibevoice` package in editable mode alongside all runtime dependencies (FastAPI, PyTorch, etc.).

3. Ensure FFmpeg is available if MP3 output is required:

   ```bash
   ffmpeg -version
   ```

## Voice Presets

Place one or more reference audio files inside `api/voices/`. Each file name (without extension) becomes the `voice` identifier for API requests.

```
api/voices/
├── en-Alice_woman.wav
├── en-Carter_man.wav
└── ...
```

Supported extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`.

> **Tip:** Files can be added while the server is running. The server rescans the directory automatically when it receives a request for an unknown voice.

## Running the Server

### Via the provided CLI wrapper

```bash
python api/server.py \
  --model_path /path/to/vibevoice-model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

Command-line options:

- `--model_path` *(required)* – directory containing the VibeVoice model checkpoints.
- `--device` – `cuda`, `mps`, `mpx`, or `cpu` (defaults to the best available device).
- `--host` – interface to bind to (`0.0.0.0` by default).
- `--port` – port number (`8000` by default).
- `--inference-steps` – diffusion inference steps (default `5`).
- `--default-cfg-scale` – default classifier-free guidance scale (default `1.3`).

### Via `uvicorn` directly

Set environment variables and run `uvicorn` manually:

```bash
export VIBEVOICE_MODEL_PATH=/path/to/vibevoice-model
export VIBEVOICE_DEVICE=cuda
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Additional environment variables: `VIBEVOICE_INFERENCE_STEPS`, `VIBEVOICE_CFG_SCALE`, `VIBEVOICE_HOST`, `VIBEVOICE_PORT`.

## API Usage

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

#### Request body

```json
{
  "model": "gpt-4o-mini-tts",
  "input": "Hello from VibeVoice!",
  "voice": "en-Alice_woman",
  "response_format": "mp3",
  "cfg_scale": 1.5,
  "speed": 1.0
}
```

- `model` – optional string, accepted for compatibility.
- `input` – required text to synthesize.
- `voice` – required voice identifier matching a file in `api/voices/` (filename without extension).
- `response_format` – optional, one of `mp3` (default), `wav`, or `flac`.
- `cfg_scale` – optional float overriding the default classifier-free guidance scale.
- `speed` – optional float accepted for compatibility; currently logged but not applied.

### Example request with `curl`

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini-tts",
        "input": "Welcome to the VibeVoice API demo.",
        "voice": "en-Alice_woman",
        "response_format": "mp3"
      }' \
  --output speech.mp3
```

### Responses

- **Success:** Streams binary audio with `Content-Type` set to the requested format and `Content-Disposition` suggesting `speech.<ext>`.
- **Errors:** Returns a JSON payload with an appropriate HTTP status code (400 for validation issues, 404 for missing voices, 500 for unexpected inference errors).

### Helper endpoints

- `GET /health` – returns `{ "status": "initializing" | "ready" }`.
- `GET /voices` – returns `{ "voices": ["voice_name", ...] }` for discovered presets.

## Notes

- The server loads the VibeVoice model on startup and keeps it in memory. Only one generation runs at a time to ensure deterministic behaviour with the current inference stack.
- The default classifier-free guidance scale can be overridden per request via `cfg_scale`.
- Speed adjustments are currently not implemented; the server logs the requested value for observability.

Happy synthesizing! 🎧
