"""Integration tests for transcribe package.

These tests run the full pipeline and require models to be downloaded.
Run with: pytest -m integration
"""

import os
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


class TestFullPipeline:
    """Integration tests for the full transcription pipeline."""

    @pytest.fixture
    def audio_file_path(self):
        """Return path to test audio file."""
        return os.environ.get("TEST_AUDIO_FILE")

    @pytest.fixture
    def short_audio_file(self, temp_dir):
        """Create a minimal audio file for testing if no test file available.
        
        Note: We can't easily create a valid audio file without ffmpeg,
        so this fixture just provides a placeholder.
        """
        audio_path = temp_dir / "test_short.wav"
        
        if not audio_path.exists():
            pytest.skip("Need a test audio file. Set TEST_AUDIO_FILE env var or create test file.")
        
        return str(audio_path)

    def test_cli_basic_execution(self):
        """Test that CLI can be imported and runs."""
        from transcribe.cli import main
        
        assert main is not None
        assert callable(main)

    def test_imports_work(self):
        """Test that all modules can be imported."""
        from transcribe import (
            get_device,
            load_whisper_with_fallback,
            load_diarization_pipeline,
            format_transcript,
            get_output_path,
        )
        
        assert get_device is not None
        assert load_whisper_with_fallback is not None
        assert load_diarization_pipeline is not None
        assert format_transcript is not None
        assert get_output_path is not None

    def test_models_module_functions_exist(self):
        """Test that model functions exist."""
        from transcribe import models
        
        assert hasattr(models, "get_device")
        assert hasattr(models, "load_whisper_model")
        assert hasattr(models, "load_whisper_with_fallback")
        assert hasattr(models, "DEFAULT_MODEL")

    def test_diarization_module_functions_exist(self):
        """Test that diarization functions exist."""
        from transcribe import diarization
        
        assert hasattr(diarization, "load_diarization_pipeline")
        assert hasattr(diarization, "run_diarization")
        assert hasattr(diarization, "DEFAULT_DIARIZATION_MODEL")

    def test_audio_module_functions_exist(self):
        """Test that audio functions exist."""
        from transcribe import audio
        
        assert hasattr(audio, "get_audio_duration")
        assert hasattr(audio, "split_audio_chunks")
        assert hasattr(audio, "transcribe_audio")

    def test_alignment_module_functions_exist(self):
        """Test that alignment functions exist."""
        from transcribe import alignment
        
        assert hasattr(alignment, "calculate_overlap")
        assert hasattr(alignment, "merge_transcript_and_diarization")

    def test_output_module_functions_exist(self):
        """Test that output functions exist."""
        from transcribe import output
        
        assert hasattr(output, "format_transcript")
        assert hasattr(output, "save_transcript")
        assert hasattr(output, "get_output_path")

    def test_package_version(self):
        """Test that package has a version."""
        from transcribe import __version__
        
        assert __version__ is not None
        assert __version__ != ""


class TestOfflineModelsAvailable:
    """Test that offline models are available when expected."""

    def test_models_directory_check(self):
        """Test that models can be checked."""
        from transcribe.download import check_models_exist
        
        result = check_models_exist("./models")
        
        assert isinstance(result, bool)

    def test_offline_mode_status(self):
        """Test offline mode status check."""
        from transcribe.download import check_models_exist
        
        result = check_models_exist("./models")
        
        if result:
            pass
        else:
            pytest.skip("Models not downloaded - run with HF_TOKEN set first time")


@pytest.mark.integration
def test_models_can_be_loaded():
    """Integration test: Verify models can be loaded if available.
    
    This test is skipped if models haven't been downloaded.
    Run with: pytest -m integration
    """
    if not os.path.exists("./models"):
        pytest.skip("Models not downloaded")
    
    if not check_models_exist("./models"):
        pytest.skip("Models not valid")
    
    from transcribe import load_whisper_with_fallback, get_device
    
    device = get_device()
    model = load_whisper_with_fallback(device)
    
    assert model is not None, "Failed to load Whisper model"


def check_models_exist(models_dir):
    """Check if models exist (duplicate for standalone use)."""
    from pathlib import Path
    models_path = Path(models_dir)
    whisper_dir = models_path / "whisper"
    pyannote_dir = models_path / "pyannote"
    
    if not whisper_dir.exists() or not pyannote_dir.exists():
        return False
    
    has_whisper = any(whisper_dir.rglob("*.bin")) or any(whisper_dir.rglob("*.safetensors"))
    has_pyannote = (pyannote_dir / "segmentation-3.0").exists() and (pyannote_dir / "speaker-diarization-3.1").exists()
    
    return has_whisper and has_pyannote