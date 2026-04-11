"""Unit tests for download module."""

import pytest
from transcribe.download import (
    check_models_exist,
    validate_pyannote_models,
    validate_whisper_model,
)


class TestCheckModelsExist:
    """Tests for check_models_exist function."""

    def test_returns_true_for_valid_models(self, mock_models_dir):
        """Should return True when all models exist."""
        result = check_models_exist(str(mock_models_dir))
        assert result is True

    def test_returns_false_for_missing_models(self, missing_models_dir):
        """Should return False when models directory is missing or empty."""
        result = check_models_exist(str(missing_models_dir))
        assert result is False

    def test_returns_false_for_invalid_path(self, temp_dir):
        """Should return False for non-existent path."""
        result = check_models_exist(str(temp_dir / "nonexistent"))
        assert result is False


class TestValidateWhisperModel:
    """Tests for validate_whisper_model function."""

    def test_valid_model(self, mock_models_dir):
        """Should return True for valid model."""
        from pathlib import Path
        model_dir = mock_models_dir / "whisper" / "openai_whisper-large-v3"
        
        result = validate_whisper_model(model_dir)
        
        assert result is True

    def test_invalid_missing_directory(self, temp_dir):
        """Should return False for missing directory."""
        result = validate_whisper_model(temp_dir / "nonexistent")
        assert result is False

    def test_invalid_missing_config(self, temp_dir):
        """Should return False when config.json is missing."""
        model_dir = temp_dir / "whisper" / "test"
        model_dir.mkdir(parents=True)
        (model_dir / "tokenizer.json").touch()
        (model_dir / "model.bin").touch()
        
        result = validate_whisper_model(model_dir)
        
        assert result is False

    def test_invalid_missing_tokenizer(self, temp_dir):
        """Should return False when tokenizer.json is missing."""
        model_dir = temp_dir / "whisper" / "test"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        (model_dir / "model.bin").touch()
        
        result = validate_whisper_model(model_dir)
        
        assert result is False

    def test_invalid_no_model_file(self, temp_dir):
        """Should return False when model file is missing."""
        model_dir = temp_dir / "whisper" / "test"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        (model_dir / "tokenizer.json").touch()
        
        result = validate_whisper_model(model_dir)
        
        assert result is False


class TestValidatePyannoteModels:
    """Tests for validate_pyannote_models function."""

    def test_valid_pyannote_models(self, mock_models_dir):
        """Should return True for valid pyannote models."""
        result = validate_pyannote_models(str(mock_models_dir))
        assert result is True

    def test_invalid_missing_directory(self, missing_models_dir):
        """Should return False when pyannote directory is missing."""
        result = validate_pyannote_models(str(missing_models_dir))
        assert result is False

    def test_invalid_missing_segmentation(self, temp_dir):
        """Should return False when segmentation directory is missing."""
        from pathlib import Path
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        
        pyannote_dir = models_dir / "pyannote"
        diarization_dir = pyannote_dir / "speaker-diarization-3.1"
        diarization_dir.mkdir(parents=True)
        (diarization_dir / "config.json").touch()
        
        result = validate_pyannote_models(str(models_dir))
        
        assert result is False

    def test_invalid_missing_diarization(self, temp_dir):
        """Should return False when diarization directory is missing."""
        from pathlib import Path
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        
        pyannote_dir = models_dir / "pyannote"
        segmentation_dir = pyannote_dir / "segmentation-3.0"
        segmentation_dir.mkdir(parents=True)
        (segmentation_dir / "config.json").touch()
        
        result = validate_pyannote_models(str(models_dir))
        
        assert result is False

    def test_empty_directories(self, temp_dir):
        """Should return False when directories are empty."""
        from pathlib import Path
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        
        pyannote_dir = models_dir / "pyannote"
        segmentation_dir = pyannote_dir / "segmentation-3.0"
        segmentation_dir.mkdir(parents=True)
        diarization_dir = pyannote_dir / "speaker-diarization-3.1"
        diarization_dir.mkdir(parents=True)
        
        result = validate_pyannote_models(str(models_dir))
        
        assert result is False


class TestConstants:
    """Tests for download module constants."""

    def test_default_whisper_model(self):
        """DEFAULT_WHISPER_MODEL should be set."""
        from transcribe.download import DEFAULT_WHISPER_MODEL
        assert DEFAULT_WHISPER_MODEL == "openai/whisper-large-v3"

    def test_diarization_model(self):
        """DIARIZATION_MODEL should be set."""
        from transcribe.download import DIARIZATION_MODEL
        assert DIARIZATION_MODEL == "pyannote/speaker-diarization-3.1"

    def test_segmentation_model(self):
        """SEGMENTATION_MODEL should be set."""
        from transcribe.download import SEGMENTATION_MODEL
        assert SEGMENTATION_MODEL == "pyannote/segmentation-3.0"