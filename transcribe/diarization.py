"""Speaker diarization using pyannote.audio."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_MODELS_DIR = "./models"


def find_local_pyannote_model(models_dir: str = DEFAULT_MODELS_DIR) -> Optional[Path]:
    """Check if pyannote models exist locally.
    
    Args:
        models_dir: Local models directory
    
    Returns:
        Path to local pyannote models if exists, None otherwise
    """
    models_path = Path(models_dir) / "pyannote"
    
    if not models_path.exists() or not models_path.is_dir():
        return None
    
    required_dirs = [
        "segmentation-3.0",
        "speaker-diarization-3.1"
    ]
    
    for dir_name in required_dirs:
        dir_path = models_path / dir_name
        if not dir_path.exists() or not any(dir_path.iterdir()):
            return None
    
    return models_path


def load_diarization_pipeline(
    hf_token: Optional[str] = None, 
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    models_dir: str = DEFAULT_MODELS_DIR
):
    """Load pyannote speaker diarization pipeline.
    
    Args:
        hf_token: HuggingFace token for model access (required if no local models)
        model_name: Diarization model name (default: pyannote/speaker-diarization-3.1)
        models_dir: Local models directory
    
    Returns:
        Pipeline instance or None on failure
    """
    from pyannote.audio import Pipeline
    import torch
    
    local_models = find_local_pyannote_model(models_dir)
    
    print("Loading speaker diarization pipeline...", file=sys.stderr)
    
    if local_models:
        print(f"Using local pyannote models: {local_models}", file=sys.stderr)
        
        diarization_dir = local_models / "speaker-diarization-3.1"
        segmentation_dir = local_models / "segmentation-3.0"
        
        if diarization_dir.exists() and segmentation_dir.exists():
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    cache_dir=str(local_models)
                )
                
                if torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
                
                return pipeline
            except Exception as e:
                print(f"Failed to load local pyannote models: {e}", file=sys.stderr)
                print("Trying online mode...", file=sys.stderr)
        else:
            print(f"Local pyannote models incomplete, trying online...", file=sys.stderr)
    
    if not hf_token:
        print("Error: HF_TOKEN required for online model download", file=sys.stderr)
        print("Please set: export HF_TOKEN=your_token", file=sys.stderr)
        return None
    
    try:
        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        
        return pipeline
    except Exception as e:
        print(f"Failed to load diarization pipeline: {e}", file=sys.stderr)
        return None


def run_diarization(pipeline, audio_path: str) -> List[Dict[str, float]]:
    """Run speaker diarization on audio file.
    
    Args:
        pipeline: Loaded pyannote Pipeline
        audio_path: Path to audio file
    
    Returns:
        List of dicts with 'start', 'end', 'speaker' keys
    """
    print(f"Running speaker diarization: {audio_path}", file=sys.stderr)
    
    try:
        diarization = pipeline(audio_path)
        
        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        print(f"Found {len(diarization_segments)} speaker segments", file=sys.stderr)
        return diarization_segments
    except Exception as e:
        print(f"Diarization failed: {e}", file=sys.stderr)
        return []
