"""Model loading utilities for Whisper transcription."""

import sys

DEFAULT_MODEL = "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1"
FALLBACK_MODEL = "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition"


def get_device() -> str:
    """Determine whether to use CUDA or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_whisper_model(model_name: str, device: str):
    """Load faster-whisper model.
    
    Args:
        model_name: HuggingFace model ID or local model path
        device: "cuda" or "cpu"
    
    Returns:
        WhisperModel instance or None on failure
    """
    from faster_whisper import WhisperModel
    
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        print(f"Loading Whisper model: {model_name}", file=sys.stderr)
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        return model
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}", file=sys.stderr)
        return None


def load_whisper_with_fallback(device: str, primary: str = DEFAULT_MODEL, fallback: str = FALLBACK_MODEL):
    """Load Whisper model with fallback on failure.
    
    Args:
        device: "cuda" or "cpu"
        primary: Primary model to try first
        fallback: Fallback model if primary fails
    
    Returns:
        WhisperModel instance or None if both fail
    """
    model = load_whisper_model(primary, device)
    
    if model is None:
        print(f"Primary model failed, trying fallback: {fallback}", file=sys.stderr)
        model = load_whisper_model(fallback, device)
    
    return model
