"""Output formatting and file writing utilities."""

import sys
from typing import List


def format_transcript(merged_segments: List[dict]) -> str:
    """Format merged segments as human-readable transcript.
    
    Args:
        merged_segments: List of dicts with 'speaker' and 'text' keys
    
    Returns:
        Formatted transcript string
    """
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
