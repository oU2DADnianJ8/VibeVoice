"""VibeVoice OpenAI-compatible text-to-speech API server."""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from pydub import AudioSegment
from pydub.exceptions import CouldntEncodeError

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


LOGGER = logging.getLogger("vibevoice.api")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

SUPPORTED_VOICE_EXTENSIONS: Tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
)

AUDIO_MEDIA_TYPES: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
}


class VoiceNotFoundError(Exception):
    """Raised when the requested voice preset is unavailable."""


class AudioEncodingError(Exception):
    """Raised when the generated audio cannot be encoded to the requested format."""


@dataclass
class SynthesisResult:
    """Represents an audio synthesis result."""

    audio_bytes: bytes
    media_type: str
    file_extension: str


@dataclass
class ServerConfig:
    """Runtime configuration for the API server."""

    model_path: Path
    device: str
    inference_steps: int = 5
    default_cfg_scale: float = 1.3


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpeechRequest(BaseModel):
    """Pydantic model that mirrors OpenAI's speech synthesis request."""

    model: Optional[str] = Field(default=None)
    text_input: str = Field(..., alias="input", description="Text to synthesize.")
    voice: str = Field(..., description="Name of the voice preset to use.")
    response_format: Optional[str] = Field(default="mp3")
    speed: Optional[float] = Field(default=None)
    cfg_scale: Optional[float] = Field(default=None, description="Classifier-free guidance scale.")

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

    @validator("text_input")
    def _validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Input text must not be empty.")
        return value

    @validator("response_format", pre=True, always=True)
    def _validate_format(cls, value: Optional[str]) -> str:
        fmt = (value or "mp3").lower()
        if fmt not in AUDIO_MEDIA_TYPES:
            raise ValueError(f"Unsupported response_format '{value}'. Supported formats: {', '.join(AUDIO_MEDIA_TYPES)}")
        return fmt

    @validator("cfg_scale")
    def _validate_cfg_scale(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("cfg_scale must be greater than zero when provided.")
        return value

    @validator("speed")
    def _validate_speed(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("speed must be greater than zero when provided.")
        return value


class VibeVoiceService:
    """Encapsulates model loading, voice management, and inference."""

    sample_rate: int = 24_000

    def __init__(
        self,
        model_path: Path,
        device: str,
        inference_steps: int = 5,
        default_cfg_scale: float = 1.3,
    ) -> None:
        self.model_path = model_path
        self.device = self._normalize_device(device)
        self.inference_steps = inference_steps
        self.default_cfg_scale = default_cfg_scale
        self.voice_directory = Path(__file__).resolve().parent / "voices"
        self.voice_cache: Dict[str, np.ndarray] = {}
        self._voice_cache_lock = threading.Lock()
        self._generation_lock = threading.Lock()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path '{self.model_path}' does not exist.")

        LOGGER.info("Loading VibeVoice model from %s", self.model_path)
        self.processor = self._load_processor()
        self.model = self._load_model()
        self.voice_presets = self._discover_voice_presets()
        if not self.voice_presets:
            LOGGER.warning(
                "No voice presets found in %s. Add .wav/.mp3/.flac files before serving requests.",
                self.voice_directory,
            )
        else:
            LOGGER.info("Discovered %d voice preset(s): %s", len(self.voice_presets), ", ".join(sorted(self._voice_display_names())))

    def _normalize_device(self, requested_device: str) -> str:
        device = (requested_device or "cpu").lower()
        if device == "mpx":
            LOGGER.info("Device 'mpx' detected. Falling back to 'mps'.")
            device = "mps"
        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                LOGGER.warning("MPS requested but not available. Falling back to CPU.")
                device = "cpu"
        if device not in {"cuda", "mps", "cpu"}:
            LOGGER.warning("Unknown device '%s'. Falling back to CPU.", device)
            device = "cpu"
        return device

    def _load_processor(self) -> VibeVoiceProcessor:
        processor = VibeVoiceProcessor.from_pretrained(str(self.model_path))
        LOGGER.info("Processor loaded successfully.")
        return processor

    def _load_model(self) -> VibeVoiceForConditionalGenerationInference:
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"

        LOGGER.info(
            "Initializing model on %s with torch_dtype=%s and attention=%s",
            self.device,
            load_dtype,
            attn_impl,
        )

        try:
            if self.device == "mps":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                model.to("mps")
            elif self.device == "cuda":
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            if attn_impl == "flash_attention_2":
                LOGGER.exception("Falling back to SDPA attention due to error: %s", exc)
                model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in {"cuda", "cpu"} else None),
                    attn_implementation="sdpa",
                )
                if self.device == "mps":
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
            LOGGER.info(
                "Language model attention implementation: %s",
                getattr(model.model.language_model.config, "_attn_implementation", "unknown"),
            )

        return model

    def _discover_voice_presets(self) -> Dict[str, Path]:
        voices: Dict[str, Path] = {}
        if not self.voice_directory.exists():
            LOGGER.warning("Voices directory %s does not exist.", self.voice_directory)
            return voices
        for audio_path in sorted(self.voice_directory.iterdir()):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in SUPPORTED_VOICE_EXTENSIONS:
                continue
            key = audio_path.stem.lower()
            if key in voices:
                LOGGER.warning("Duplicate voice preset '%s' detected. Keeping the first occurrence only.", key)
                continue
            voices[key] = audio_path
        return voices

    def _voice_display_names(self) -> Iterable[str]:
        for path in self.voice_presets.values():
            yield path.stem

    def refresh_voice_presets(self) -> None:
        """Re-scan the voices directory for new or removed presets."""
        self.voice_presets = self._discover_voice_presets()

    def list_available_voices(self) -> List[str]:
        return sorted(path.stem for path in self.voice_presets.values())

    def _resolve_voice(self, voice_name: str) -> Tuple[str, Path]:
        if not voice_name:
            raise ValueError("Voice name must not be empty.")
        normalized = Path(voice_name).stem.lower()
        if normalized not in self.voice_presets:
            self.refresh_voice_presets()
        if normalized not in self.voice_presets:
            available = ", ".join(self.list_available_voices()) or "<none>"
            raise VoiceNotFoundError(
                f"Voice '{voice_name}' not found. Available voices: {available}."
            )
        return normalized, self.voice_presets[normalized]

    def _load_voice_sample(self, voice_key: str, voice_path: Path) -> np.ndarray:
        with self._voice_cache_lock:
            if voice_key in self.voice_cache:
                return self.voice_cache[voice_key]

        try:
            audio, sr = sf.read(str(voice_path))
        except Exception as exc:  # pragma: no cover - I/O failure
            raise RuntimeError(f"Failed to load audio for voice '{voice_path.name}': {exc}") from exc

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError(f"Voice preset '{voice_path.name}' contains no audio samples.")

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        with self._voice_cache_lock:
            self.voice_cache[voice_key] = audio
        return audio

    def synthesize_speech(
        self,
        text: str,
        voice_name: str,
        response_format: str,
        cfg_scale: Optional[float] = None,
    ) -> SynthesisResult:
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty after trimming whitespace.")

        voice_key, voice_path = self._resolve_voice(voice_name)
        voice_sample = self._load_voice_sample(voice_key, voice_path)

        formatted_script = self._format_script(text)
        LOGGER.debug("Formatted script for synthesis: %s", formatted_script)

        inputs = self.processor(
            text=[formatted_script],
            voice_samples=[[voice_sample]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        target_device = self.device if self.device in {"cuda", "mps"} else "cpu"
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device)

        cfg_value = cfg_scale if cfg_scale is not None else self.default_cfg_scale
        LOGGER.info(
            "Generating audio (voice=%s, cfg_scale=%.2f, format=%s)",
            voice_path.stem,
            cfg_value,
            response_format,
        )

        with self._generation_lock:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_value,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    refresh_negative=True,
                    verbose=False,
                )

        if not output.speech_outputs or output.speech_outputs[0] is None:
            raise RuntimeError("The model did not return any speech output.")

        audio_tensor = output.speech_outputs[0]
        if torch.is_tensor(audio_tensor):
            audio = audio_tensor.detach().cpu().float().numpy()
        else:
            audio = np.asarray(audio_tensor, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.squeeze()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("Generated audio is empty.")

        audio = np.clip(audio, -1.0, 1.0)

        buffer, media_type, extension = self._encode_audio(audio, response_format)
        return SynthesisResult(audio_bytes=buffer.getvalue(), media_type=media_type, file_extension=extension)

    def _format_script(self, text: str) -> str:
        if "speaker" in text.lower() and ":" in text:
            return text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = [text.strip()]
        formatted_lines = [f"Speaker 0: {line}" for line in lines]
        return "\n".join(formatted_lines)

    def _encode_audio(self, audio: np.ndarray, response_format: str) -> Tuple[io.BytesIO, str, str]:
        fmt = response_format.lower()
        if fmt not in AUDIO_MEDIA_TYPES:
            raise ValueError(f"Unsupported response format: {response_format}")

        buffer = io.BytesIO()
        media_type = AUDIO_MEDIA_TYPES[fmt]

        if fmt in {"wav", "flac"}:
            try:
                sf.write(buffer, audio, self.sample_rate, format=fmt)
            except Exception as exc:  # pragma: no cover - encoding failure
                raise AudioEncodingError(f"Failed to encode audio as {fmt}: {exc}") from exc
        else:  # mp3
            int16_audio = self._float_to_int16(audio)
            segment = AudioSegment(
                int16_audio.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=int16_audio.dtype.itemsize,
                channels=1,
            )
            try:
                segment.export(buffer, format="mp3")
            except CouldntEncodeError as exc:  # pragma: no cover - missing ffmpeg
                raise AudioEncodingError(
                    "Failed to encode MP3 audio. Ensure that FFmpeg is installed and available in PATH."
                ) from exc
        buffer.seek(0)
        return buffer, media_type, fmt

    @staticmethod
    def _float_to_int16(audio: np.ndarray) -> np.ndarray:
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def shutdown(self) -> None:
        """Release model resources."""
        LOGGER.info("Shutting down VibeVoice service.")
        self.voice_cache.clear()
        self.model = None
        self.processor = None


def load_default_config() -> ServerConfig:
    model_path = Path(os.getenv("VIBEVOICE_MODEL_PATH", "./vibevoice-model"))
    device = os.getenv("VIBEVOICE_DEVICE", _default_device())
    inference_steps = int(os.getenv("VIBEVOICE_INFERENCE_STEPS", "5"))
    default_cfg_scale = float(os.getenv("VIBEVOICE_CFG_SCALE", "1.3"))
    return ServerConfig(
        model_path=model_path,
        device=device,
        inference_steps=inference_steps,
        default_cfg_scale=default_cfg_scale,
    )


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="VibeVoice TTS API", version="1.0.0")
    app.state.config = config
    app.state.service: Optional[VibeVoiceService] = None

    @app.on_event("startup")
    async def startup_event() -> None:
        LOGGER.info("Starting VibeVoice API server with model at %s", config.model_path)
        try:
            service = await asyncio.to_thread(
                VibeVoiceService,
                config.model_path,
                config.device,
                config.inference_steps,
                config.default_cfg_scale,
            )
        except Exception as exc:
            LOGGER.exception("Failed to initialize VibeVoice service: %s", exc)
            raise
        app.state.service = service

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        service = getattr(app.state, "service", None)
        if service is not None:
            await asyncio.to_thread(service.shutdown)

    async def get_service() -> VibeVoiceService:
        service = getattr(app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded yet.")
        return service

    @app.get("/health")
    async def health() -> Dict[str, str]:
        service = getattr(app.state, "service", None)
        status_text = "ready" if service is not None else "initializing"
        return {"status": status_text}

    @app.get("/voices")
    async def list_voices(service: VibeVoiceService = Depends(get_service)) -> Dict[str, List[str]]:
        return {"voices": service.list_available_voices()}

    @app.post("/v1/audio/speech")
    async def create_speech(
        request: SpeechRequest,
        service: VibeVoiceService = Depends(get_service),
    ) -> StreamingResponse:
        if request.speed not in (None, 1.0):
            LOGGER.info("Speed parameter provided (%.2f) but is currently not applied.", request.speed)

        try:
            result = await asyncio.to_thread(
                service.synthesize_speech,
                request.text_input,
                request.voice,
                request.response_format,
                request.cfg_scale,
            )
        except VoiceNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except (ValueError, AudioEncodingError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("Model inference failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected failure
            LOGGER.exception("Unexpected error during synthesis: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error.") from exc

        async def audio_stream() -> AsyncIterator[bytes]:
            yield result.audio_bytes

        response = StreamingResponse(audio_stream(), media_type=result.media_type)
        response.headers["Content-Disposition"] = f'attachment; filename="speech.{result.file_extension}"'
        response.headers["X-Model"] = str(request.model or "VibeVoice")
        response.headers["X-Sample-Rate"] = str(service.sample_rate)
        return response

    return app


app = create_app(load_default_config())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VibeVoice OpenAI-compatible TTS server.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(os.getenv("VIBEVOICE_MODEL_PATH", "./vibevoice-model")),
        help="Path to the VibeVoice model directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("VIBEVOICE_DEVICE", _default_device()),
        choices=["cpu", "cuda", "mps", "mpx"],
        help="Computation device to use.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("VIBEVOICE_HOST", "0.0.0.0"),
        help="Host interface for the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VIBEVOICE_PORT", "8000")),
        help="Port for the server.",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=int(os.getenv("VIBEVOICE_INFERENCE_STEPS", "5")),
        help="Number of diffusion inference steps.",
    )
    parser.add_argument(
        "--default-cfg-scale",
        type=float,
        default=float(os.getenv("VIBEVOICE_CFG_SCALE", "1.3")),
        help="Default classifier-free guidance scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ServerConfig(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        default_cfg_scale=args.default_cfg_scale,
    )

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise SystemExit("uvicorn must be installed to run the server.") from exc

    uvicorn.run(
        create_app(config),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
