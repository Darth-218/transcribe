#!/usr/bin/env python3
"""
Arabic-English Audio Transcription with Speaker Diarization

This script transcribes mixed Arabic-English audio with speaker diarization using:
- Whisper: oddadmix/MasriSwitch-Gemma3n-Transcriber-v1 (fallback: MohamedRashad/Arabic-Whisper-CodeSwitching-Edition)
- Diarization: pyannote/speaker-diarization-3.1

Requirements:
- HF_TOKEN environment variable for HuggingFace model access
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def get_device():
    """Determine whether to use CUDA or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_whisper_model(model_name: str, device: str):
    """Load faster-whisper model with fallback support."""
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


def load_diarization_pipeline(hf_token: str):
    """Load pyannote speaker diarization pipeline."""
    from pyannote.audio import Pipeline
    import torch
    
    print("Loading speaker diarization pipeline...", file=sys.stderr)
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        
        return pipeline
    except Exception as e:
        print(f"Failed to load diarization pipeline: {e}", file=sys.stderr)
        return None


def transcribe_audio(model, audio_path: str, language: str = "ar"):
    """Transcribe audio file using faster-whisper."""
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


def run_diarization(pipeline, audio_path: str):
    """Run speaker diarization on audio file."""
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


def calculate_overlap(seg_start: float, seg_end: float, 
                      dia_start: float, dia_end: float) -> float:
    """Calculate overlap duration between two time ranges."""
    return max(0.0, min(seg_end, dia_end) - max(seg_start, dia_start))


def merge_transcript_and_diarization(transcript_segments: list, 
                                      diarization_segments: list) -> list:
    """Merge transcript segments with speaker labels based on temporal overlap."""
    if not transcript_segments:
        return []
    
    if not diarization_segments:
        return [{"speaker": "SPEAKER_00", **seg} for seg in transcript_segments]
    
    diarization_sorted = sorted(diarization_segments, key=lambda x: x["start"])
    
    merged = []
    for seg in transcript_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        speaker_overlap = {}
        for dia in diarization_sorted:
            overlap = calculate_overlap(
                seg_start, seg_end,
                dia["start"], dia["end"]
            )
            if overlap > 0:
                speaker = dia["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap
        
        if speaker_overlap:
            speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]
        else:
            speaker = "SPEAKER_00"
        
        merged.append({
            "speaker": speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"]
        })
    
    return merged


def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        import librosa
        return librosa.get_duration(path=audio_path)


def split_audio_chunks(audio_path: str, chunk_duration: int = 1800):
    """Split audio into chunks if longer than chunk_duration (default 30 min)."""
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


def process_chunk(model, pipeline, audio_path: str, 
                  start_time: float, end_time: float) -> list:
    """Process a single chunk of audio."""
    import librosa
    import soundfile as sf
    import tempfile
    import os
    
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


def format_transcript(merged_segments: list) -> str:
    """Format merged segments as human-readable transcript."""
    lines = []
    
    current_speaker = None
    for seg in merged_segments:
        speaker = seg["speaker"]
        text = seg["text"]
        
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"{speaker}: {text}")
            current_speaker = speaker
        else:
            lines.append(text)
    
    return "\n".join(lines)


def save_transcript(transcript: str, output_path: str):
    """Save transcript to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"Transcript saved to: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to save transcript: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Arabic-English audio with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s /path/to/mentorship_session.mp3
  
Environment:
  HF_TOKEN    HuggingFace token for model access (required)
"""
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (wav, mp3, etc.)"
    )
    parser.add_argument(
        "--model",
        default="oddadmix/MasriSwitch-Gemma3n-Transcriber-v1",
        help="Whisper model to use (default: oddadmix/MasriSwitch-Gemma3n-Transcriber-v1)"
    )
    parser.add_argument(
        "--fallback-model",
        default="MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        help="Fallback Whisper model if primary fails"
    )
    parser.add_argument(
        "--language",
        default="ar",
        help="Language code for transcription (default: ar for Arabic)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=1800,
        help="Chunk duration in seconds for long audio (default: 1800 = 30 min)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: <input>_transcript.txt)"
    )
    
    args = parser.parse_args()
    
    audio_file = args.audio_file
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is required", file=sys.stderr)
        print("Please set: export HF_TOKEN=your_huggingface_token", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60, file=sys.stderr)
    print("Arabic-English Audio Transcription with Speaker Diarization", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    device = get_device()
    print(f"Using device: {device}", file=sys.stderr)
    
    print(f"Primary model: {args.model}", file=sys.stderr)
    model = load_whisper_model(args.model, device)
    
    if model is None:
        print(f"Primary model failed, trying fallback: {args.fallback_model}", file=sys.stderr)
        model = load_whisper_model(args.fallback_model, device)
        
        if model is None:
            print("Error: Failed to load both primary and fallback models", file=sys.stderr)
            sys.exit(1)
    
    pipeline = load_diarization_pipeline(hf_token)
    if pipeline is None:
        print("Error: Failed to load diarization pipeline", file=sys.stderr)
        sys.exit(1)
    
    chunks = split_audio_chunks(audio_file, args.chunk_duration)
    
    if len(chunks) == 1:
        print("Processing single audio file...", file=sys.stderr)
        transcript_segments, _ = transcribe_audio(model, audio_file)
        diarization_segments = run_diarization(pipeline, audio_file)
        merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
    else:
        print(f"Processing {len(chunks)} chunks for long audio...", file=sys.stderr)
        all_merged = []
        for start, end, path in chunks:
            chunk_result = process_chunk(model, pipeline, path, start, end)
            all_merged.extend(chunk_result)
        merged = all_merged
    
    if not merged:
        print("Error: No transcription output", file=sys.stderr)
        sys.exit(1)
    
    transcript = format_transcript(merged)
    
    print("\n" + "=" * 60, file=sys.stderr)
    print("TRANSCRIPT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(transcript)
    print("=" * 60, file=sys.stderr)
    
    if args.output:
        output_path = args.output
    else:
        input_path = Path(audio_file)
        output_path = str(input_path.with_suffix(".txt")).replace(
            input_path.suffix, "_transcript.txt"
        )
    
    save_transcript(transcript, output_path)
    
    print(f"\nTranscription complete!", file=sys.stderr)


if __name__ == "__main__":
    main()