#!/usr/bin/env python3
"""Command-line interface for the transcription tool."""

import argparse
import os
import sys
import tempfile
import warnings

# Suppress third-party warnings that are harmless but noisy
warnings.filterwarnings("ignore", "The MPEG_LAYER_III")
warnings.filterwarnings("ignore", "degrees of freedom")
warnings.filterwarnings("ignore", "std(): degrees of freedom")

from tqdm import tqdm
from transcribe import models, diarization, audio, alignment, output
from transcribe.models import DEFAULT_MODEL, FALLBACK_MODEL


def process_chunk_realtime(model, pipeline, audio_path, start_time, end_time, pbar, transcription_only=False):
    """Process a single chunk with real-time segment progress updates.
    
    Args:
        model: Whisper model
        pipeline: Diarization pipeline (None if transcription_only)
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        pbar: tqdm progress bar to update
        transcription_only: If True, skip diarization
    
    Returns:
        List of merged transcript segments
    """
    import soundfile as sf
    
    y, sr = audio.load_audio_chunk(audio_path, start_time, end_time, sr=16000)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    sf.write(tmp_path, y, sr)
    
    try:
        transcript_segments = []
        for segment in audio.transcribe_audio(model, tmp_path):
            segment["start"] += start_time
            segment["end"] += start_time
            transcript_segments.append(segment)
            pbar.update(1)
        
        if not transcription_only and pipeline is not None:
            diarization_segments = diarization.run_diarization(pipeline, tmp_path)
            
            for dia in diarization_segments:
                dia["start"] += start_time
                dia["end"] += start_time
            
            merged = alignment.merge_transcript_and_diarization(transcript_segments, diarization_segments)
        else:
            merged = transcript_segments
        
        return merged
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Arabic-English audio with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s /path/to/mentorship_session.mp3
  %(prog)s audio.mp3 -t

Options:
  -t, --transcription-only   Run only transcription (skip speaker diarization)
                             No HF_TOKEN required in this mode

Environment:
  HF_TOKEN    HuggingFace token (required only for diarization mode)
"""
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (wav, mp3, etc.)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        help=f"Fallback Whisper model if primary fails (default: {FALLBACK_MODEL})"
    )
    parser.add_argument(
        "--language",
        default="ar",
        help="Language code for transcription (default: ar for Arabic)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=300,
        help="Chunk duration in seconds for long audio (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: <input>_transcript.txt)"
    )
    parser.add_argument(
        "--transcription-only", "-t",
        action="store_true",
        help="Run only transcription (skip speaker diarization)"
    )
    
    args = parser.parse_args()
    
    audio_file = args.audio_file
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)
    
    print("Arabic-English Audio Transcription", file=sys.stderr)
    
    device = models.get_device()
    print(f"Using device: {device}", file=sys.stderr)
    
    print("Loading models...", file=sys.stderr)
    model = models.load_whisper_with_fallback(
        device,
        primary=args.model,
        fallback=args.fallback_model
    )
    
    if model is None:
        print("Error: Failed to load both primary and fallback models", file=sys.stderr)
        sys.exit(1)
    
    pipeline = None
    if not args.transcription_only:
        print("Mode: Transcription + Diarization", file=sys.stderr)
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Error: HF_TOKEN environment variable is required for diarization", file=sys.stderr)
            print("Please set: export HF_TOKEN=your_huggingface_token", file=sys.stderr)
            print("Or use -t flag for transcription only", file=sys.stderr)
            sys.exit(1)
        
        pipeline = diarization.load_diarization_pipeline(hf_token)
        if pipeline is None:
            print("Error: Failed to load diarization pipeline", file=sys.stderr)
            sys.exit(1)
    else:
        print("Mode: Transcription only (no diarization)", file=sys.stderr)
    
    chunks = audio.split_audio_chunks(audio_file, args.chunk_duration)
    
    if len(chunks) == 1:
        desc = "Transcribing (no diarization)" if args.transcription_only else "Transcribing audio"
        with tqdm(desc=desc, unit="seg", file=sys.stderr) as pbar:
            transcript_segments = []
            for segment in audio.transcribe_audio(model, audio_file, args.language):
                transcript_segments.append(segment)
                pbar.update(1)
        
        if args.transcription_only:
            merged = transcript_segments
        else:
            with tqdm(desc="Running diarization", unit="turn", file=sys.stderr) as pbar:
                diarization_segments = diarization.run_diarization(pipeline, audio_file)
                pbar.update(1)
                pbar.set_postfix({"turns": len(diarization_segments)})
            
            merged = alignment.merge_transcript_and_diarization(transcript_segments, diarization_segments)
    else:
        with tqdm(desc="Processing chunks", unit="seg", file=sys.stderr) as pbar:
            all_merged = []
            for i, (start, end, path) in enumerate(chunks):
                pbar.set_description(f"Chunk {i+1}/{len(chunks)}")
                chunk_result = process_chunk_realtime(
                    model, pipeline, path, start, end, pbar,
                    transcription_only=args.transcription_only
                )
                all_merged.extend(chunk_result)
            merged = all_merged
    
    if not merged:
        print("Error: No transcription output", file=sys.stderr)
        sys.exit(1)
    
    transcript = output.format_transcript(merged, transcription_only=args.transcription_only)
    
    print("\n" + "=" * 60, file=sys.stderr)
    print("TRANSCRIPT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(transcript)
    print("=" * 60, file=sys.stderr)
    
    output_path = output.get_output_path(audio_file, args.output)
    output.save_transcript(transcript, output_path)
    
    print(f"\nTranscription complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
