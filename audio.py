"""Audio loading and chunking utilities."""

import sys
import os
import tempfile
from typing import List, Tuple

import numpy as np


def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        import librosa
        return librosa.get_duration(path=audio_path)


def split_audio_chunks(audio_path: str, chunk_duration: int = 1800) -> List[Tuple[float, float, str]]:
    """Split audio into chunks if longer than chunk_duration.
    
    Args:
        audio_path: Path to audio file
        chunk_duration: Maximum chunk duration in seconds (default: 30 min)
    
    Returns:
        List of (start_time, end_time, audio_path) tuples
    """
    duration = get_audio_duration(audio_path)
    
    if duration <= chunk_duration:
        return [(0.0, duration, audio_path)]
    
    chunks = []
    num_chunks = int(np.ceil(duration / chunk_duration))
    
    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)
        chunks.append((start, end, audio_path))
    
    return chunks


def process_chunk(model, pipeline, audio_path: str, start_time: float, end_time: float) -> List[dict]:
    """Process a single chunk of audio.
    
    Args:
        model: Whisper model
        pipeline: Diarization pipeline
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        List of merged transcript segments with speaker labels
    """
    import librosa
    import soundfile as sf
    
    from transcribe.models import transcribe_audio
    from transcribe.diarization import run_diarization
    from transcribe.alignment import merge_transcript_and_diarization
    
    print(f"Processing chunk: {start_time:.1f}s to {end_time:.1f}s", file=sys.stderr)
    
    y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time - start_time)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    sf.write(tmp_path, y, sr)
    
    try:
        transcript_segments, _ = transcribe_audio(model, tmp_path)
        
        diarization_segments = run_diarization(pipeline, tmp_path)
        
        offset = start_time
        for seg in transcript_segments:
            seg["start"] += offset
            seg["end"] += offset
        
        for dia in diarization_segments:
            dia["start"] += offset
            dia["end"] += offset
        
        merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
        
        return merged
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def transcribe_audio(model, audio_path: str, language: str = "ar") -> tuple:
    """Transcribe audio file using faster-whisper.
    
    Args:
        model: Loaded WhisperModel
        audio_path: Path to audio file
        language: Language code (default: ar for Arabic)
    
    Returns:
        Tuple of (transcript_segments list, info object)
    """
    print(f"Transcribing: {audio_path}", file=sys.stderr)
    
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        transcript_segments = []
        for seg in segments:
            transcript_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
        
        print(f"Transcribed {len(transcript_segments)} segments", file=sys.stderr)
        return transcript_segments, info
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        return [], None
