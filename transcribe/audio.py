"""Audio loading and chunking utilities."""

import subprocess
import sys
import os
import tempfile
from typing import List, Tuple, Generator, Any

import numpy as np


def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds using ffprobe.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path
    ], capture_output=True, text=True)
    
    try:
        duration = float(result.stdout.strip())
        return duration
    except ValueError:
        return 0.0


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


def load_audio_chunk(audio_path: str, start_time: float, end_time: float, sr: int = 16000) -> tuple:
    """Load audio chunk using ffmpeg directly.
    
    This avoids librosa/soundfile MP3 warnings.
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        sr: Sample rate (default: 16000)
    
    Returns:
        Tuple of (audio array, sample rate)
    """
    duration = end_time - start_time
    
    result = subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-f", "f32le",
        "-ar", str(sr),
        "-ac", "1",
        "-ss", str(start_time),
        "-t", str(duration),
        "-loglevel", "quiet",
        "pipe:1"
    ], capture_output=True)
    
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, sr


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
    import soundfile as sf
    
    from transcribe.diarization import run_diarization
    from transcribe.alignment import merge_transcript_and_diarization
    
    y, sr = load_audio_chunk(audio_path, start_time, end_time, sr=16000)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    sf.write(tmp_path, y, sr)
    
    try:
        transcript_segments = list(transcribe_audio(model, tmp_path))
        
        if pipeline is not None:
            diarization_segments = run_diarization(pipeline, tmp_path)
        else:
            diarization_segments = []
        
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


def transcribe_audio(model, audio_path: str, language: str = "ar") -> Generator[dict, None, None]:
    """Transcribe audio file using faster-whisper as a generator.
    
    Yields segments as they are processed for real-time progress updates.
    
    Args:
        model: Loaded WhisperModel
        audio_path: Path to audio file
        language: Language code (default: ar for Arabic)
    
    Yields:
        dict with 'start', 'end', 'text' keys for each segment
    """
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        for seg in segments:
            yield {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            }
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
