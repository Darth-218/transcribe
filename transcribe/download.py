"""Download and cache models locally for offline use."""

import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_WHISPER_MODEL = "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"


def ensure_models_dir(models_dir: str) -> Path:
    """Ensure models directory exists.
    
    Args:
        models_dir: Base models directory path
    
    Returns:
        Path object for models directory
    """
    path = Path(models_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_whisper_model(model_id: str, models_dir: str, hf_token: Optional[str] = None):
    """Download Whisper model to local directory.
    
    Args:
        model_id: HuggingFace model ID (e.g., "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1")
        models_dir: Target directory for models
        hf_token: HuggingFace token for gated models
    """
    from faster_whisper import WhisperModel
    
    target_dir = Path(models_dir) / "whisper" / model_id.replace("/", "_")
    
    if target_dir.exists() and (target_dir / "model.bin").exists():
        print(f"Whisper model already exists: {target_dir}", file=sys.stderr)
        return target_dir
    
    print(f"Downloading Whisper model: {model_id}", file=sys.stderr)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    model = WhisperModel(model_id, device="cpu", compute_type="int8")
    
    print(f"Whisper model downloaded to: {target_dir}", file=sys.stderr)
    return target_dir


def download_pyannote_models(models_dir: str, hf_token: str):
    """Download pyannote models to local directory.
    
    Args:
        models_dir: Target directory for models
        hf_token: HuggingFace token (required for pyannote models)
    """
    from huggingface_hub import snapshot_download
    
    target_dir = Path(models_dir) / "pyannote"
    
    if target_dir.exists() and (target_dir / "speaker-diarization-3.1").exists():
        print(f"Pyannote models already exist: {target_dir}", file=sys.stderr)
        return target_dir
    
    print(f"Downloading pyannote models...", file=sys.stderr)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            SEGMENTATION_MODEL,
            cache_dir=str(target_dir / "segmentation"),
            local_dir=str(target_dir / "segmentation-3.0"),
            token=hf_token,
        )
    except Exception as e:
        print(f"Warning: Could not download segmentation model: {e}", file=sys.stderr)
    
    try:
        snapshot_download(
            DIARIZATION_MODEL,
            cache_dir=str(target_dir / "diarization"),
            local_dir=str(target_dir / "speaker-diarization-3.1"),
            token=hf_token,
        )
    except Exception as e:
        print(f"Warning: Could not download diarization model: {e}", file=sys.stderr)
    
    print(f"Pyannote models downloaded to: {target_dir}", file=sys.stderr)
    return target_dir


def check_models_exist(models_dir: str) -> bool:
    """Check if all required models are downloaded.
    
    Args:
        models_dir: Models directory path
    
    Returns:
        True if all models exist, False otherwise
    """
    models_path = Path(models_dir)
    
    whisper_dir = models_path / "whisper"
    if not whisper_dir.exists() or not list(whisper_dir.rglob("*.bin")):
        return False
    
    pyannote_dir = models_path / "pyannote"
    if not pyannote_dir.exists():
        return False
    
    return True


def download_all(models_dir: str = "./models", hf_token: Optional[str] = None):
    """Download all required models for offline use.
    
    Args:
        models_dir: Target directory for models (default: "./models")
        hf_token: HuggingFace token (required for first download)
    """
    if hf_token is None:
        print("Error: HF_TOKEN required for initial model download", file=sys.stderr)
        print("Please set: export HF_TOKEN=your_token", file=sys.stderr)
        sys.exit(1)
    
    ensure_models_dir(models_dir)
    
    print("=" * 60, file=sys.stderr)
    print("Downloading models for offline use", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    try:
        download_whisper_model(DEFAULT_WHISPER_MODEL, models_dir, hf_token)
    except Exception as e:
        print(f"Warning: Could not download Whisper model: {e}", file=sys.stderr)
        print("Falling back to online mode for Whisper", file=sys.stderr)
    
    try:
        download_pyannote_models(models_dir, hf_token)
    except Exception as e:
        print(f"Warning: Could not download pyannote models: {e}", file=sys.stderr)
        print("Falling back to online mode for pyannote", file=sys.stderr)
    
    print("=" * 60, file=sys.stderr)
    print("Model download complete!", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for offline use")
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Target directory for models (default: ./models)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()
    
    hf_token = args.token or os.environ.get("HF_TOKEN")
    download_all(args.models_dir, hf_token)
