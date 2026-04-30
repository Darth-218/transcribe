"""Output formatting and file writing utilities."""

import sys
from typing import List


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS timestamp.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string (e.g., "02:35")
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_transcript(merged_segments: List[dict], transcription_only: bool = False) -> str:
    """Format merged segments as human-readable transcript.
    
    Args:
        merged_segments: List of dicts with 'speaker'/'start'/'end'/'text' keys
        transcription_only: If True, show timestamps instead of speaker labels
    
    Returns:
        Formatted transcript string
    """
    lines = []
    
    if transcription_only:
        for seg in merged_segments:
            start_time = format_time(seg.get("start", 0))
            end_time = format_time(seg.get("end", 0))
            text = seg["text"]
            lines.append(f"[{start_time} - {end_time}] {text}")
    else:
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
    """Save transcript to file.
    
    Args:
        transcript: Formatted transcript string
        output_path: Path to output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"Transcript saved to: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to save transcript: {e}", file=sys.stderr)


def get_output_path(audio_file: str, custom_output: str | None = None) -> str:
    """Get output file path.
    
    Args:
        audio_file: Input audio file path
        custom_output: Custom output path if provided
    
    Returns:
        Output file path
    """
    from pathlib import Path
    
    if custom_output is not None:
        return custom_output
    
    input_path = Path(audio_file)
    output_path = str(input_path.with_suffix(".txt")).replace(
        input_path.suffix, "_transcript.txt"
    )
    
    return output_path
