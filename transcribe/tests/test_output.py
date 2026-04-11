"""Unit tests for output module."""

import os
import tempfile

import pytest
from transcribe.output import format_transcript, get_output_path, save_transcript


class TestFormatTranscript:
    """Tests for format_transcript function."""

    def test_empty_segments(self):
        """Empty segments should return empty string."""
        result = format_transcript([])
        assert result == ""

    def test_single_segment(self):
        """Single segment should be formatted correctly."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello world"}
        ]
        
        result = format_transcript(segments)
        
        expected = "SPEAKER_00: Hello world"
        assert result == expected

    def test_multiple_same_speaker(self):
        """Multiple segments from same speaker."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello"},
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 10.0, "text": "world"},
        ]
        
        result = format_transcript(segments)
        
        expected = "SPEAKER_00: Hello\nworld"
        assert result == expected

    def test_multiple_speakers(self):
        """Multiple speakers should add blank line between."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "Hi there"},
        ]
        
        result = format_transcript(segments)
        
        expected = "SPEAKER_00: Hello\n\nSPEAKER_01: Hi there"
        assert result == expected

    def test_three_speakers(self):
        """Three different speakers."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "First"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "Second"},
            {"speaker": "SPEAKER_02", "start": 10.0, "end": 15.0, "text": "Third"},
        ]
        
        result = format_transcript(segments)
        
        lines = result.split("\n")
        assert lines[0] == "SPEAKER_00: First"
        assert lines[1] == ""
        assert lines[2] == "SPEAKER_01: Second"
        assert lines[3] == ""
        assert lines[4] == "SPEAKER_02: Third"

    def test_arabic_text(self):
        """Arabic text should be handled correctly."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "مرحبا"}
        ]
        
        result = format_transcript(segments)
        
        assert "مرحبا" in result

    def test_preserves_text_content(self):
        """Text content should be exactly preserved."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "  Hello  "}
        ]
        
        result = format_transcript(segments)
        
        assert "Hello  " in result


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_default_wav(self):
        """Output path for WAV file."""
        result = get_output_path("audio.wav")
        assert result == "audio_transcript.txt"

    def test_default_mp3(self):
        """Output path for MP3 file."""
        result = get_output_path("audio.mp3")
        assert result == "audio_transcript.txt"

    def test_custom_output(self):
        """Custom output path should be used."""
        result = get_output_path("audio.wav", "custom.txt")
        assert result == "custom.txt"

    def test_custom_with_path(self):
        """Custom output with directory."""
        result = get_output_path("audio.wav", "/tmp/output.txt")
        assert result == "/tmp/output.txt"

    def test_strips_extension(self):
        """Should strip original extension correctly."""
        result = get_output_path("audio.long.mp3")
        assert ".mp3" not in result

    def test_no_extension(self):
        """File without extension."""
        result = get_output_path("audiofile")
        assert result == "audiofile_transcript.txt"


class TestSaveTranscript:
    """Tests for save_transcript function."""

    def test_save_to_file(self, temp_dir):
        """Should save transcript to file."""
        output_path = temp_dir / "output.txt"
        
        save_transcript("Test transcript", str(output_path))
        
        assert output_path.exists()
        assert output_path.read_text() == "Test transcript"

    def test_overwrite_existing(self, temp_dir):
        """Should overwrite existing file."""
        output_path = temp_dir / "output.txt"
        output_path.write_text("Old content")
        
        save_transcript("New content", str(output_path))
        
        assert output_path.read_text() == "New content"

    def test_creates_parent_dirs(self, temp_dir):
        """Should create parent directories if needed."""
        output_path = temp_dir / "subdir" / "output.txt"
        
        save_transcript("Content", str(output_path))
        
        assert output_path.exists()
        assert output_path.read_text() == "Content"

    def test_unicode_content(self, temp_dir):
        """Should save unicode content correctly."""
        output_path = temp_dir / "output.txt"
        
        save_transcript("مرحبا Hello", str(output_path))
        
        assert output_path.read_text() == "مرحبا Hello"