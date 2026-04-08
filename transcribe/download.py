"""Download and cache models locally for offline use."""

import os
import sys
from pathlib import Path
from typing import Optional, List

DEFAULT_WHISPER_MODEL = "whisper-large-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"

WHISPER_REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
]

WHISPER_OPTIONAL_FILES = [
    "model.bin",
    "model.safetensors",
    "tokenizer_config.json",
    "vocabulary.json",
    "preprocessor_config.json",
]


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


def get_whisper_model_files(models_path: Path) -> List[Path]:
    """Get all Whisper model files (both .bin and .safetensors).
    
    Args:
        models_path: Path to models directory
    
    Returns:
        List of model file paths
    """
    bin_files = list(models_path.rglob("*.bin"))
    safetensors_files = list(models_path.rglob("*.safetensors"))
    return bin_files + safetensors_files


def validate_whisper_model(model_dir: Path) -> bool:
    """Validate that all required Whisper model files exist.
    
    Args:
        model_dir: Path to model directory
    
    Returns:
        True if model is valid, False otherwise
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    
    for required_file in WHISPER_REQUIRED_FILES:
        if not (model_dir / required_file).exists():
            return False
    
    has_model_file = get_whisper_model_files(model_dir)
    return len(has_model_file) > 0


def download_whisper_model(model_id: str, models_dir: str, hf_token: Optional[str] = None):
    """Download Whisper model to local directory.
    
    Uses huggingface_hub snapshot_download to properly download all model files.
    
    Args:
        model_id: HuggingFace model ID (e.g., "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1")
        models_dir: Target directory for models
        hf_token: HuggingFace token for gated models
    """
    from huggingface_hub import snapshot_download
    
    target_dir = Path(models_dir) / "whisper" / model_id.replace("/", "_")
    
    if validate_whisper_model(target_dir):
        print(f"Whisper model already exists: {target_dir}", file=sys.stderr)
        return target_dir
    
    print(f"Downloading Whisper model: {model_id}", file=sys.stderr)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            token=hf_token,
            ignore_patterns=["*.md", "*.txt", ".github/*"]
        )
        
        if validate_whisper_model(target_dir):
            print(f"Whisper model downloaded to: {target_dir}", file=sys.stderr)
            _list_model_files(target_dir)
        else:
            print(f"Warning: Model downloaded but validation failed", file=sys.stderr)
        
        return target_dir
    except Exception as e:
        print(f"Error downloading Whisper model: {e}", file=sys.stderr)
        raise


def _list_model_files(model_dir: Path):
    """List all files in the model directory for debugging."""
    print(f"Model files in {model_dir}:", file=sys.stderr)
    for f in sorted(model_dir.iterdir()):
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name}: {size} bytes", file=sys.stderr)


def get_pyannote_model_structure(models_path: Path) -> dict:
    """Get pyannote model subdirectory structure.
    
    Args:
        models_path: Path to pyannote models directory
    
    Returns:
        Dict with model names and their paths
    """
    structure = {}
    
    if not models_path.exists():
        return structure
    
    for item in models_path.iterdir():
        if item.is_dir():
            structure[item.name] = item
    
    return structure


def validate_pyannote_models(models_dir: str) -> bool:
    """Validate that pyannote models are properly downloaded.
    
    Args:
        models_dir: Path to models directory
    
    Returns:
        True if pyannote models are valid
    """
    pyannote_dir = Path(models_dir) / "pyannote"
    
    if not pyannote_dir.exists():
        return False
    
    structure = get_pyannote_model_structure(pyannote_dir)
    
    required_models = [
        "segmentation-3.0",
        "speaker-diarization-3.1"
    ]
    
    for model_name in required_models:
        if model_name not in structure:
            return False
        
        model_path = structure[model_name]
        
        if not any(model_path.iterdir()):
            return False
    
    return True


def download_pyannote_models(models_dir: str, hf_token: str):
    """Download pyannote models to local directory.
    
    Args:
        models_dir: Target directory for models
        hf_token: HuggingFace token (required for pyannote models)
    """
    from huggingface_hub import snapshot_download
    
    target_dir = Path(models_dir) / "pyannote"
    
    if validate_pyannote_models(models_dir):
        print(f"Pyannote models already exist: {target_dir}", file=sys.stderr)
        return target_dir
    
    print(f"Downloading pyannote models...", file=sys.stderr)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        segmentation_dir = target_dir / "segmentation-3.0"
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=SEGMENTATION_MODEL,
            local_dir=str(segmentation_dir),
            token=hf_token,
            ignore_patterns=["*.md", "*.txt", ".github/*"]
        )
        print(f"Downloaded segmentation model", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not download segmentation model: {e}", file=sys.stderr)
    
    try:
        diarization_dir = target_dir / "speaker-diarization-3.1"
        diarization_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=DIARIZATION_MODEL,
            local_dir=str(diarization_dir),
            token=hf_token,
            ignore_patterns=["*.md", "*.txt", ".github/*"]
        )
        print(f"Downloaded diarization model", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not download diarization model: {e}", file=sys.stderr)
    
    if validate_pyannote_models(models_dir):
        print(f"Pyannote models downloaded to: {target_dir}", file=sys.stderr)
    else:
        print(f"Warning: Pyannote models downloaded but validation failed", file=sys.stderr)
    
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
    if not validate_whisper_model(whisper_dir):
        return False
    
    if not validate_pyannote_models(models_dir):
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
