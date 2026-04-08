"""Model loading utilities for Whisper transcription."""

import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1"
FALLBACK_MODEL = "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition"
DEFAULT_MODELS_DIR = "./models"


def get_device() -> str:
    """Determine whether to use CUDA or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def find_local_model(model_id: str, models_dir: str = DEFAULT_MODELS_DIR) -> Optional[Path]:
    """Check if model exists locally.
    
    Args:
        model_id: HuggingFace model ID (e.g., "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1")
        models_dir: Local models directory
    
    Returns:
        Path to local model if exists, None otherwise
    """
    models_path = Path(models_dir)
    local_name = model_id.replace("/", "_")
    local_path = models_path / "whisper" / local_name
    
    if local_path.exists() and list(local_path.glob("*.bin")):
        return local_path
    
    return None


def load_whisper_model(
    model_name: str, 
    device: str,
    models_dir: str = DEFAULT_MODELS_DIR,
    hf_token: Optional[str] = None
):
    """Load faster-whisper model.
    
    Args:
        model_name: HuggingFace model ID or local model path
        device: "cuda" or "cpu"
        models_dir: Local models directory for caching
        hf_token: HuggingFace token for gated models
    
    Returns:
        WhisperModel instance or None on failure
    """
    from faster_whisper import WhisperModel
    
    compute_type = "float16" if device == "cuda" else "int8"
    
    local_model_path = find_local_model(model_name, models_dir)
    
    if local_model_path:
        print(f"Loading Whisper model from local: {local_model_path}", file=sys.stderr)
        try:
            model = WhisperModel(
                str(local_model_path),
                device=device,
                compute_type=compute_type
            )
            return model
        except Exception as e:
            print(f"Failed to load local model: {e}, trying online", file=sys.stderr)
    
    print(f"Loading Whisper model: {model_name}", file=sys.stderr)
    
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        return model
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}", file=sys.stderr)
        return None


def load_whisper_with_fallback(
    device: str, 
    primary: str = DEFAULT_MODEL, 
    fallback: str = FALLBACK_MODEL,
    models_dir: str = DEFAULT_MODELS_DIR,
    hf_token: Optional[str] = None
):
    """Load Whisper model with fallback on failure.
    
    Args:
        device: "cuda" or "cpu"
        primary: Primary model to try first
        fallback: Fallback model if primary fails
        models_dir: Local models directory
        hf_token: HuggingFace token for gated models
    
    Returns:
        WhisperModel instance or None if both fail
    """
    model = load_whisper_model(primary, device, models_dir, hf_token)
    
    if model is None:
        print(f"Primary model failed, trying fallback: {fallback}", file=sys.stderr)
        model = load_whisper_model(fallback, device, models_dir, hf_token)
    
    return model
