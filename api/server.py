"""VibeVoice Text-to-Speech API server."""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from pydub import AudioSegment
from pydub.exceptions import CouldntEncodeError

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.getLogger("vibevoice.api.server")

# Maps requested response formats to (<media type>, <file extension>)
SUPPORTED_RESPONSE_FORMATS: Dict[str, Tuple[str, str]] = {
    "mp3": ("audio/mpeg", "mp3"),
    "wav": ("audio/wav", "wav"),
    "flac": ("audio/flac", "flac"),
}


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class VoiceNotFoundError(Exception):
    """Raised when a requested voice preset cannot be located."""


class AudioConversionError(Exception):
    """Raised when generated audio fails to convert to the requested format."""


@dataclass
class ServerConfig:
    """Configuration used to bootstrap the FastAPI application."""

    model_path: str
    device: str
    host: str
    port: int
    voices_dir: Path
    output_dir: Path
    default_cfg_scale: float
    inference_steps: int


def _sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return sanitized or "speech"


def _chunk_bytes(data: bytes, chunk_size: int = 64 * 1024) -> Iterable[bytes]:
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


def _persist_mp3(output_dir: Path, audio_bytes: bytes, voice_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    identifier = uuid4().hex[:8]
    safe_voice = _sanitize_for_filename(voice_name)
    filename = f"{timestamp}_{safe_voice}_{identifier}.mp3"
    output_path = output_dir / filename

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as mp3_file:
        mp3_file.write(audio_bytes)

    return output_path


class VoiceLibrary:
    """Manages loading and caching of voice presets from disk."""

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    )

    def __init__(self, directory: Path, target_sr: int) -> None:
        self.directory = directory
        self.target_sr = target_sr
        self.voice_paths: Dict[str, Path] = {}
        self._voice_audio: Dict[str, np.ndarray] = {}
        self._name_map: Dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        """Refresh the voice preset cache by scanning the voices directory."""
        if not self.directory.exists():
            raise RuntimeError(
                f"Voices directory '{self.directory}' does not exist. "
                "Add reference audio files before starting the server."
            )

        voice_paths: Dict[str, Path] = {}
        voice_audio: Dict[str, np.ndarray] = {}
        name_map: Dict[str, str] = {}

        for audio_path in sorted(self.directory.iterdir()):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            name = audio_path.stem
            audio_array = self._read_audio(audio_path)
            if audio_array.size == 0:
                logger.warning("Skipping voice preset '%s' (empty or unreadable)", audio_path.name)
                continue

            voice_paths[name] = audio_path
            voice_audio[name] = audio_array
            name_map[name.lower()] = name

        if not voice_paths:
            raise RuntimeError(
                f"No voice presets were found in '{self.directory}'. "
                "Add audio files with supported extensions (.wav, .mp3, .flac, .ogg, .m4a, .aac)."
            )

        self.voice_paths = voice_paths
        self._voice_audio = voice_audio
        self._name_map = name_map

        logger.info("Loaded %d voice preset(s) from %s", len(self.voice_paths), self.directory)
        logger.info("Available voices: %s", ", ".join(sorted(self.voice_paths)))

    def _read_audio(self, audio_path: Path) -> np.ndarray:
        """Load an audio file and resample it to the target sample rate."""
        try:
            audio, sr = sf.read(str(audio_path))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            return np.asarray(audio, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - we want to log all failures
            logger.error("Failed to load voice preset '%s': %s", audio_path.name, exc)
            return np.array([], dtype=np.float32)

    def get(self, name: str) -> np.ndarray:
        """Return the cached audio array for a voice preset."""
        canonical = self._name_map.get(name.lower())
        if canonical is None:
            raise VoiceNotFoundError(f"Voice preset '{name}' was not found.")
        return self._voice_audio[canonical]

    def list_voices(self) -> List[str]:
        """Return the list of available voice preset identifiers."""
        return sorted(self.voice_paths.keys())


class VibeVoiceTTS:
    """Wrapper around the VibeVoice model that performs speech synthesis."""

    def __init__(
        self,
        model_path: str,
        device: str,
        voices_dir: Path,
        inference_steps: int = 5,
        default_cfg_scale: float = 1.3,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.default_cfg_scale = default_cfg_scale
        self.processor: Optional[VibeVoiceProcessor] = None
        self.model: Optional[VibeVoiceForConditionalGenerationInference] = None
        self.sample_rate: int = 24000
        self.voice_library: Optional[VoiceLibrary] = None
        self._generation_lock = threading.Lock()

        self._load_model()
        self.voice_library = VoiceLibrary(voices_dir, target_sr=self.sample_rate)

    @staticmethod
    def _mps_available() -> bool:
        return _mps_is_available()

    def _normalize_device(self, device: str) -> str:
        normalized = device.lower()
        if normalized == "mpx":
            logger.warning("Device 'mpx' detected, normalizing to 'mps'.")
            normalized = "mps"
        if normalized == "mps" and not self._mps_available():
            logger.warning("MPS device requested but not available. Falling back to CPU.")
            normalized = "cpu"
        return normalized

    def _load_model(self) -> None:
        logger.info("Loading processor and model from %s", self.model_path)

        normalized_device = self._normalize_device(self.device)
        self.device = normalized_device

        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        audio_processor = getattr(self.processor, "audio_processor", None)
        if audio_processor is not None and hasattr(audio_processor, "sampling_rate"):
            self.sample_rate = int(audio_processor.sampling_rate)
        else:
            self.sample_rate = 24000

        if normalized_device == "mps":
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        elif normalized_device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"

        logger.info(
            "Using device=%s torch_dtype=%s attention=%s",
            normalized_device,
            load_dtype,
            attn_impl_primary,
        )

        try:
            if normalized_device == "mps":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                model.to("mps")
            elif normalized_device == "cuda":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as exc:  # noqa: BLE001 - load fallback on any failure
            if attn_impl_primary == "flash_attention_2":
                logger.error("Failed to load model with flash_attention_2: %s", exc)
                logger.info("Retrying with sdpa attention implementation.")
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(normalized_device if normalized_device in {"cuda", "cpu"} else None),
                    attn_implementation="sdpa",
                )
                if normalized_device == "mps":
                    model.to("mps")
            else:
                raise

        model.eval()
        model.model.noise_scheduler = model.model.noise_scheduler.from_config(
            model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        if hasattr(model.model, "language_model"):
            attn_impl = getattr(
                model.model.language_model.config,
                "_attn_implementation",
                "unknown",
            )
            logger.info("Language model attention implementation: %s", attn_impl)

        self.model = model

    def available_voices(self) -> List[str]:
        if not self.voice_library:
            return []
        return self.voice_library.list_voices()

    def synthesize_waveform(
        self,
        text: str,
        voice_name: str,
        cfg_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        if not text.strip():
            raise ValueError("Input text must not be empty.")
        if not self.voice_library:
            raise RuntimeError("Voice library has not been initialized.")

        voice_audio = self.voice_library.get(voice_name)
        formatted_text = self._format_text(text)
        cfg_scale = cfg_scale if cfg_scale is not None else self.default_cfg_scale
        if cfg_scale <= 0:
            raise ValueError("cfg_scale must be greater than zero.")

        inputs = self.processor(
            text=[formatted_text],
            voice_samples=[[voice_audio]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        target_device = self.device if self.device in {"cuda", "mps"} else "cpu"
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device)

        assert self.model is not None  # for type checkers

        with self._generation_lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    refresh_negative=True,
                )

        speech_outputs = getattr(outputs, "speech_outputs", None)
        if not speech_outputs or speech_outputs[0] is None:
            raise RuntimeError("Model did not return any audio output.")

        audio = speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            waveform_tensor = audio.detach()
            if waveform_tensor.dtype != torch.float32:
                waveform_tensor = waveform_tensor.to(torch.float32)
            waveform = waveform_tensor.cpu().numpy()
        else:
            waveform = np.asarray(audio, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        if waveform.size == 0:
            raise RuntimeError("Generated audio is empty.")

        return waveform.astype(np.float32), self.sample_rate

    def synthesize_to_format(
        self,
        text: str,
        voice_name: str,
        response_format: str,
        cfg_scale: Optional[float] = None,
    ) -> Tuple[bytes, str, str]:
        waveform, sample_rate = self.synthesize_waveform(text, voice_name, cfg_scale=cfg_scale)
        return self._convert_audio(waveform, sample_rate, response_format)

    def _format_text(self, text: str) -> str:
        normalized = text.replace("’", "'").strip()
        if not normalized:
            raise ValueError("Input text must not be empty after normalization.")
        return f"Speaker 0: {normalized}"

    def _convert_audio(self, waveform: np.ndarray, sample_rate: int, response_format: str) -> Tuple[bytes, str, str]:
        fmt = response_format.lower()
        if fmt not in SUPPORTED_RESPONSE_FORMATS:
            raise ValueError(
                f"Unsupported response_format '{response_format}'. Supported values: {', '.join(SUPPORTED_RESPONSE_FORMATS)}"
            )

        media_type, extension = SUPPORTED_RESPONSE_FORMATS[fmt]
        clipped = np.clip(waveform, -1.0, 1.0)

        if fmt in {"wav", "flac"}:
            buffer = io.BytesIO()
            sf.write(buffer, clipped.astype(np.float32), sample_rate, format=fmt.upper())
            buffer.seek(0)
            return buffer.read(), extension, media_type

        if fmt == "mp3":
            try:
                segment = self._to_audio_segment(clipped, sample_rate)
                buffer = io.BytesIO()
                segment.export(buffer, format="mp3")
                buffer.seek(0)
                return buffer.read(), extension, media_type
            except CouldntEncodeError as exc:  # pragma: no cover - depends on ffmpeg availability
                raise AudioConversionError(
                    "Failed to encode MP3 audio. Ensure FFmpeg is installed and available in PATH."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - log unexpected conversion failures
                raise AudioConversionError(f"Failed to convert audio to MP3: {exc}") from exc

        raise AudioConversionError(f"Unhandled response format: {response_format}")

    def _to_audio_segment(self, waveform: np.ndarray, sample_rate: int) -> AudioSegment:
        int16_audio = np.int16(waveform * 32767.0)
        return AudioSegment(
            data=int16_audio.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )


class SpeechRequest(BaseModel):
    """Request body for the OpenAI-compatible TTS endpoint."""

    model: Optional[str] = Field(default=None, description="Model identifier (ignored, kept for compatibility).")
    input: str = Field(..., description="Text content to synthesize.")
    voice: str = Field(..., description="Voice preset name to use for synthesis.")
    response_format: Optional[str] = Field(default="mp3", description="Audio format of the response.")
    stream: Optional[bool] = Field(default=True, description="Whether the response should use HTTP streaming.")
    speed: Optional[float] = Field(default=None, description="Playback speed multiplier (currently unused).")
    cfg_scale: Optional[float] = Field(default=None, description="Override the default CFG scale for synthesis.")

    @validator("input")
    def _validate_input(cls, value: str) -> str:  # noqa: D401 - short validator doc
        if not value or not value.strip():
            raise ValueError("input must not be empty.")
        return value

    @validator("voice")
    def _validate_voice(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("voice must not be empty.")
        return value

    @validator("response_format")
    def _validate_response_format(cls, value: Optional[str]) -> str:
        return (value or "mp3").lower()


app = FastAPI(title="VibeVoice Text-to-Speech API", version="1.0.0")
_startup_config: Optional[ServerConfig] = None


def configure_app(config: ServerConfig) -> None:
    """Register configuration that should be applied during startup."""
    global _startup_config  # noqa: PLW0603 - module-level configuration is intentional
    _startup_config = config


def _auto_select_device(device: str) -> str:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_is_available():
            return "mps"
        return "cpu"
    return requested


def _config_from_env() -> ServerConfig:
    base_dir = Path(__file__).resolve().parent
    model_path = os.environ.get("VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-1.5b")
    device = _auto_select_device(os.environ.get("VIBEVOICE_DEVICE", "auto"))
    host = os.environ.get("VIBEVOICE_HOST", "0.0.0.0")
    port = int(os.environ.get("VIBEVOICE_PORT", "8000"))
    cfg_scale = float(os.environ.get("VIBEVOICE_CFG_SCALE", "1.3"))
    inference_steps = int(os.environ.get("VIBEVOICE_INFERENCE_STEPS", "5"))
    voices_dir_env = os.environ.get("VIBEVOICE_VOICES_DIR")
    voices_dir = Path(voices_dir_env) if voices_dir_env else base_dir / "voices"
    output_dir_env = os.environ.get("VIBEVOICE_OUTPUT_DIR")
    output_dir = Path(output_dir_env) if output_dir_env else base_dir / "output"

    return ServerConfig(
        model_path=model_path,
        device=device,
        host=host,
        port=port,
        voices_dir=voices_dir,
        output_dir=output_dir,
        default_cfg_scale=cfg_scale,
        inference_steps=inference_steps,
    )


@app.on_event("startup")
async def _startup_event() -> None:
    config = _startup_config or _config_from_env()
    app.state.config = config
    config.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        app.state.tts = VibeVoiceTTS(
            model_path=config.model_path,
            device=config.device,
            voices_dir=config.voices_dir,
            inference_steps=config.inference_steps,
            default_cfg_scale=config.default_cfg_scale,
        )
    except Exception as exc:  # noqa: BLE001 - surface initialization errors
        logger.exception("Failed to initialize VibeVoice TTS engine: %s", exc)
        raise


def _get_tts_engine() -> VibeVoiceTTS:
    tts = getattr(app.state, "tts", None)
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS engine is not ready.")
    return tts


def _resolve_cfg_scale(requested: Optional[float], default_value: float) -> float:
    if requested is None:
        return default_value
    if requested <= 0:
        raise HTTPException(status_code=400, detail="cfg_scale must be greater than zero.")
    return requested


@app.options("/v1/audio/speech")
async def options_speech(request: Request) -> Response:
    """Handle CORS preflight requests for the speech synthesis endpoint."""

    origin = request.headers.get("origin") or "*"
    request_headers = request.headers.get("access-control-request-headers")

    headers = {
        "Allow": "OPTIONS, POST",
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "OPTIONS, POST",
    }

    if request_headers:
        headers["Access-Control-Allow-Headers"] = request_headers
    else:
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

    return Response(status_code=204, headers=headers)


@app.post("/v1/audio/speech")
async def create_speech(payload: SpeechRequest, tts: VibeVoiceTTS = Depends(_get_tts_engine)) -> Response:
    response_format = (payload.response_format or "mp3").lower()
    streaming_enabled = True if payload.stream is None else bool(payload.stream)
    cfg_scale = _resolve_cfg_scale(payload.cfg_scale, app.state.config.default_cfg_scale)

    logger.info(
        "Received synthesis request - model=%s voice=%s format=%s cfg_scale=%.2f speed=%s streaming=%s",
        payload.model,
        payload.voice,
        response_format,
        cfg_scale,
        payload.speed,
        streaming_enabled,
    )

    try:
        waveform, sample_rate = await run_in_threadpool(
            tts.synthesize_waveform,
            payload.input,
            payload.voice,
            cfg_scale,
        )
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to synthesize speech.") from exc
    except Exception as exc:  # noqa: BLE001 - catch all synthesis errors
        logger.exception("Unexpected error during synthesis: %s", exc)
        raise HTTPException(status_code=500, detail="Unexpected error during synthesis.") from exc

    try:
        audio_bytes, extension, media_type = await run_in_threadpool(
            tts._convert_audio,
            waveform,
            sample_rate,
            response_format,
        )
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AudioConversionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        if response_format == "mp3":
            mp3_bytes = audio_bytes
        else:
            mp3_bytes, _, _ = await run_in_threadpool(
                tts._convert_audio,
                waveform,
                sample_rate,
                "mp3",
            )
    except AudioConversionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        saved_path = await run_in_threadpool(
            _persist_mp3,
            app.state.config.output_dir,
            mp3_bytes,
            payload.voice,
        )
        logger.info("Saved synthesized audio to %s", saved_path)
    except Exception as exc:  # noqa: BLE001 - file system errors should be surfaced
        logger.exception("Failed to persist generated audio: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist generated audio.") from exc

    headers = {
        "Content-Disposition": f'attachment; filename="speech.{extension}"',
    }
    if streaming_enabled:
        return StreamingResponse(_chunk_bytes(audio_bytes), media_type=media_type, headers=headers)

    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@app.get("/health", summary="Health check")
async def health() -> Dict[str, object]:
    """Return basic service status information."""
    tts = getattr(app.state, "tts", None)
    voices = tts.available_voices() if tts else []
    return {
        "status": "ok" if tts else "initializing",
        "voices": voices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VibeVoice TTS API server.")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1.5b", help="Path or identifier of the VibeVoice model.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto, cuda, mps, cpu).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind the server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose the API.")
    parser.add_argument("--voices_dir", type=Path, default=Path(__file__).resolve().parent / "voices", help="Directory containing voice preset audio files.")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "output", help="Directory to store generated MP3 files.")
    parser.add_argument("--cfg_scale", type=float, default=1.3, help="Default CFG scale to use for synthesis.")
    parser.add_argument("--inference_steps", type=int, default=5, help="Number of diffusion inference steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    device = _auto_select_device(args.device)
    config = ServerConfig(
        model_path=args.model_path,
        device=device,
        host=args.host,
        port=args.port,
        voices_dir=args.voices_dir,
        output_dir=args.output_dir,
        default_cfg_scale=args.cfg_scale,
        inference_steps=args.inference_steps,
    )

    configure_app(config)

    import uvicorn

    logger.info(
        "Starting server on %s:%d with model '%s' (device=%s)",
        config.host,
        config.port,
        config.model_path,
        config.device,
    )

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
