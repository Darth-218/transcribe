"""Unit tests for alignment module."""

import pytest
from transcribe.alignment import calculate_overlap, merge_transcript_and_diarization


class TestCalculateOverlap:
    """Tests for calculate_overlap function."""

    def test_no_overlap(self):
        """Segments with no overlap should return 0."""
        result = calculate_overlap(0, 5, 10, 15)
        assert result == 0.0

    def test_full_overlap(self):
        """Completely overlapping segments should return full duration."""
        result = calculate_overlap(0, 5, 0, 5)
        assert result == 5.0

    def test_partial_overlap(self):
        """Partially overlapping segments should return overlap duration."""
        result = calculate_overlap(0, 5, 3, 8)
        assert result == 2.0

    def test_partial_overlap_reversed(self):
        """Partial overlap in reversed order."""
        result = calculate_overlap(3, 8, 0, 5)
        assert result == 2.0

    def test_contained_overlap(self):
        """One segment contained in another."""
        result = calculate_overlap(0, 10, 3, 7)
        assert result == 4.0

    def test_adjacent_no_overlap(self):
        """Adjacent segments with no overlap."""
        result = calculate_overlap(0, 5, 5, 10)
        assert result == 0.0


class TestMergeTranscriptAndDiarization:
    """Tests for merge_transcript_and_diarization function."""

    def test_no_diarization(self, sample_transcript_segments):
        """When no diarization, should use default SPEAKER_00."""
        result = merge_transcript_and_diarization(sample_transcript_segments, [])
        
        assert len(result) == len(sample_transcript_segments)
        assert all(seg["speaker"] == "SPEAKER_00" for seg in result)

    def test_no_transcript(self):
        """When no transcript, should return empty list."""
        result = merge_transcript_and_diarization([], [{"start": 0, "end": 5, "speaker": "SPEAKER_00"}])
        assert result == []

    def test_single_transcript_single_speaker(self, sample_transcript_segments):
        """Single transcript with single speaker diarization."""
        diarization = [{"start": 0, "end": 15, "speaker": "SPEAKER_00"}]
        
        result = merge_transcript_and_diarization(sample_transcript_segments, diarization)
        
        assert len(result) == 3
        assert all(seg["speaker"] == "SPEAKER_00" for seg in result)

    def test_multiple_speakers(self, sample_transcript_segments):
        """Multiple speakers across segments."""
        diarization = [
            {"start": 0, "end": 6, "speaker": "SPEAKER_00"},
            {"start": 6, "end": 12, "speaker": "SPEAKER_01"},
            {"start": 12, "end": 15, "speaker": "SPEAKER_00"},
        ]
        
        result = merge_transcript_and_diarization(sample_transcript_segments, diarization)
        
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"
        assert result[2]["speaker"] == "SPEAKER_00"

    def test_overlap_selection(self, sample_transcript_segments):
        """Speaker with most overlap should be selected."""
        diarization = [
            {"start": 2, "end": 7, "speaker": "SPEAKER_01"},
            {"start": 4, "end": 6, "speaker": "SPEAKER_00"},
        ]
        
        result = merge_transcript_and_diarization(sample_transcript_segments, diarization)
        
        assert "text" in result[0]
        assert "speaker" in result[0]

    def test_preserves_transcript_text(self, sample_transcript_segments):
        """Should preserve original transcript text."""
        diarization = [{"start": 0, "end": 15, "speaker": "SPEAKER_00"}]
        
        result = merge_transcript_and_diarization(sample_transcript_segments, diarization)
        
        assert result[0]["text"] == "Hello world"
        assert result[1]["text"] == "This is a test"
        assert result[2]["text"] == "Thank you"

    def test_preserves_timing(self, sample_transcript_segments):
        """Should preserve original timing."""
        diarization = [{"start": 0, "end": 15, "speaker": "SPEAKER_00"}]
        
        result = merge_transcript_and_diarization(sample_transcript_segments, diarization)
        
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 5.0