"""Arabic-English Audio Transcription with Speaker Diarization.

This package provides tools for transcribing mixed Arabic-English audio
with speaker diarization using faster-whisper and pyannote.audio.

Usage:
    python -m transcribe audio.wav
    python transcribe/cli.py audio.wav
"""

from transcribe.models import (
    get_device,
    load_whisper_model,
    load_whisper_with_fallback,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
)
from transcribe.diarization import (
    load_diarization_pipeline,
    run_diarization,
    DEFAULT_DIARIZATION_MODEL,
)
from transcribe.audio import (
    get_audio_duration,
    split_audio_chunks,
    process_chunk,
    transcribe_audio,
)
from transcribe.alignment import (
    calculate_overlap,
    merge_transcript_and_diarization,
)
from transcribe.output import (
    format_transcript,
    save_transcript,
    get_output_path,
)

__version__ = "0.1.0"
