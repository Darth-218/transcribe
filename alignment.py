"""Alignment of transcript segments with speaker labels."""

from typing import List, Dict


def calculate_overlap(seg_start: float, seg_end: float, 
                      dia_start: float, dia_end: float) -> float:
    """Calculate overlap duration between two time ranges.
    
    Args:
        seg_start, seg_end: Transcript segment time range
        dia_start, dia_end: Diarization segment time range
    
    Returns:
        Overlap duration in seconds
    """
    return max(0.0, min(seg_end, dia_end) - max(seg_start, dia_start))


def merge_transcript_and_diarization(transcript_segments: list, 
                                      diarization_segments: list) -> list:
    """Merge transcript segments with speaker labels based on temporal overlap.
    
    Args:
        transcript_segments: List of transcript dicts with 'start', 'end', 'text'
        diarization_segments: List of diarization dicts with 'start', 'end', 'speaker'
    
    Returns:
        List of merged dicts with 'speaker', 'start', 'end', 'text'
    """
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
