"""Shared pytest fixtures for transcribe tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file path."""
    audio_path = temp_dir / "test_audio.wav"
    audio_path.write_text("mock audio content")
    return str(audio_path)


@pytest.fixture
def sample_transcript_segments():
    """Sample transcript segments for testing."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Hello world"},
        {"start": 5.5, "end": 10.0, "text": "This is a test"},
        {"start": 10.5, "end": 15.0, "text": "Thank you"},
    ]


@pytest.fixture
def sample_diarization_segments():
    """Sample diarization segments for testing."""
    return [
        {"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00"},
        {"start": 8.0, "end": 15.0, "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def sample_merged_segments():
    """Sample merged segments with speaker labels."""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello world"},
        {"speaker": "SPEAKER_00", "start": 5.5, "end": 10.0, "text": "This is a test"},
        {"speaker": "SPEAKER_01", "start": 10.5, "end": 15.0, "text": "Thank you"},
    ]


@pytest.fixture
def mock_whisper_model():
    """Create a mock Whisper model."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_diarization_pipeline():
    """Create a mock diarization pipeline."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_models_dir(temp_dir):
    """Create a mock models directory structure."""
    models_dir = temp_dir / "models"
    whisper_dir = models_dir / "whisper" / "openai_whisper-large-v3"
    whisper_dir.mkdir(parents=True)
    
    (whisper_dir / "config.json").write_text('{}')
    (whisper_dir / "tokenizer.json").write_text('{}')
    (whisper_dir / "model-00001-of-00002.safetensors").write_text("x" * 1000)
    
    pyannote_dir = models_dir / "pyannote"
    segmentation_dir = pyannote_dir / "segmentation-3.0"
    segmentation_dir.mkdir(parents=True)
    (segmentation_dir / "config.json").write_text('{}')
    
    diarization_dir = pyannote_dir / "speaker-diarization-3.1"
    diarization_dir.mkdir(parents=True)
    (diarization_dir / "config.json").write_text('{}')
    
    return models_dir


@pytest.fixture
def missing_models_dir(temp_dir):
    """Create a directory with missing models."""
    models_dir = temp_dir / "models"
    models_dir.mkdir(parents=True)
    return models_dir


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset module imports after each test."""
    import sys
    modules_to_remove = [k for k in sys.modules if k.startswith("transcribe.")]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)
    yield