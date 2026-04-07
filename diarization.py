"""Speaker diarization using pyannote.audio."""

import sys
from typing import List, Dict

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def load_diarization_pipeline(hf_token: str, model_name: str = DEFAULT_DIARIZATION_MODEL):
    """Load pyannote speaker diarization pipeline.
    
    Args:
        hf_token: HuggingFace token for model access
        model_name: Diarization model name (default: pyannote/speaker-diarization-3.1)
    
    Returns:
        Pipeline instance or None on failure
    """
    from pyannote.audio import Pipeline
    import torch
    
    print("Loading speaker diarization pipeline...", file=sys.stderr)
    
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
