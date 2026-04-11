"""Unit tests for models module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from transcribe.models import (
    DEFAULT_MODEL,
    DEFAULT_MODELS_DIR,
    FALLBACK_MODEL,
    get_device,
    get_whisper_model_files,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_string(self):
        """Should return a string."""
        result = get_device()
        assert isinstance(result, str)

    def test_cuda_when_available(self):
        """Should return cuda when available."""
        with patch("torch.cuda.is_available", return_value=True):
            result = get_device()
            assert result == "cuda"

    def test_cpu_when_cuda_not_available(self):
        """Should return cpu when CUDA not available."""
        with patch("torch.cuda.is_available", return_value=False):
            result = get_device()
            assert result == "cpu"

    def test_cpu_on_exception(self):
        """Should return cpu on exception."""
        with patch("torch.cuda.is_available", side_effect=Exception("No CUDA")):
            result = get_device()
            assert result == "cpu"


class TestConstants:
    """Tests for module constants."""

    def test_default_model(self):
        """DEFAULT_MODEL should be set."""
        assert DEFAULT_MODEL == "large-v3"

    def test_fallback_model(self):
        """FALLBACK_MODEL should be set."""
        assert FALLBACK_MODEL == "medium"

    def test_default_models_dir(self):
        """DEFAULT_MODELS_DIR should be set."""
        assert DEFAULT_MODELS_DIR == "./models"


class TestGetWhisperModelFiles:
    """Tests for get_whisper_model_files function."""

    def test_no_bin_files(self, temp_dir):
        """Should return empty list when no model files."""
        result = get_whisper_model_files(temp_dir)
        assert result == []

    def test_finds_bin_files(self, temp_dir):
        """Should find .bin files."""
        (temp_dir / "model.bin").touch()
        
        result = get_whisper_model_files(temp_dir)
        
        assert len(result) == 1

    def test_finds_safetensors_files(self, temp_dir):
        """Should find .safetensors files."""
        (temp_dir / "model-00001-of-00002.safetensors").touch()
        
        result = get_whisper_model_files(temp_dir)
        
        assert len(result) == 1

    def test_finds_both_types(self, temp_dir):
        """Should find both .bin and .safetensors files."""
        (temp_dir / "model.bin").touch()
        (temp_dir / "model.safetensors").touch()
        
        result = get_whisper_model_files(temp_dir)
        
        assert len(result) == 2

    def test_reursive_search(self, temp_dir):
        """Should search recursively."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "model.bin").touch()
        
        result = get_whisper_model_files(temp_dir)
        
        assert len(result) == 1


class TestFindLocalModel:
    """Tests for find_local_model function."""

    def test_directory_not_exists(self, temp_dir):
        """Should return None when directory doesn't exist."""
        from transcribe.models import find_local_model
        
        result = find_local_model("large-v3", str(temp_dir))
        
        assert result is None

    def test_valid_model(self, mock_models_dir):
        """Should find valid model."""
        from transcribe.models import find_local_model
        
        result = find_local_model("openai_whisper-large-v3", str(mock_models_dir))
        
        assert result is not None
        assert "whisper" in str(result)

    def test_missing_config(self, temp_dir):
        """Should return None when config.json is missing."""
        from transcribe.models import find_local_model
        
        model_dir = temp_dir / "whisper" / "test_model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").touch()
        
        result = find_local_model("test_model", str(temp_dir))
        
        assert result is None

    def test_missing_tokenizer(self, temp_dir):
        """Should return None when tokenizer.json is missing."""
        from transcribe.models import find_local_model
        
        model_dir = temp_dir / "whisper" / "test_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        
        result = find_local_model("test_model", str(temp_dir))
        
        assert result is None

    def test_wrong_model_name(self, temp_dir):
        """Should return None for non-existent model."""
        from transcribe.models import find_local_model
        
        result = find_local_model("non_existent_model", str(temp_dir))
        
        assert result is None