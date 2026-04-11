# Development Documentation

## Project Overview

Arabic-English audio transcription with speaker diarization using faster-whisper and pyannote.audio.

### Current Model Configuration

| Purpose | Model ID | Notes |
|---------|---------|-------|
| Transcription | `openai/whisper-large-v3` | Download name |
| Transcription Runtime | `large-v3` | faster-whisper shorthand |
| Fallback Runtime | `medium` | Smaller fallback |
| Diarization | `pyannote/speaker-diarization-3.1` | Speaker identification |

### Why Two Model Names?

faster-whisper uses shorthand names at runtime, but huggingface_hub needs full repo IDs for downloading:

| Download (huggingface_hub) | Runtime (faster-whisper) |
|---------------------------|------------------------|
| `openai/whisper-large-v3` | `large-v3` |
| `openai/whisper-medium` | `medium` |

This is handled in the code:
- `download.py`: Uses full repo IDs for downloading
- `models.py`: Uses short names for loading from local cache

## Project Structure

```
transcribe/
├── __init__.py         # Package exports
├── __main__.py         # python -m transcribe entry
├── cli.py             # CLI argument parsing + main()
├── models.py          # Whisper model loading + detection
├── diarization.py     # pyannote pipeline loading
├── audio.py          # Audio chunking + loading
├── alignment.py       # Merge transcripts with speakers
├── output.py        # Formatting + file output
└── download.py      # Model download utilities
```

### Module Responsibilities

| Module | Functions |
|--------|-----------|
| `models.py` | `get_device()`, `find_local_model()`, `load_whisper_model()`, `load_whisper_with_fallback()` |
| `diarization.py` | `find_local_pyannote_model()`, `load_diarization_pipeline()`, `run_diarization()` |
| `audio.py` | `get_audio_duration()`, `split_audio_chunks()`, `process_chunk()`, `transcribe_audio()` |
| `alignment.py` | `calculate_overlap()`, `merge_transcript_and_diarization()` |
| `output.py` | `format_transcript()`, `save_transcript()`, `get_output_path()` |
| `download.py` | `download_whisper_model()`, `download_pyannote_models()`, `check_models_exist()`, `download_all()` |

### Nix Configuration

`flake.nix` provides:
- Python 3.11 with required packages
- Auto-download of models on first run (if HF_TOKEN is set)
- Status display showing offline readiness

## Offline Mode

### First-Time Setup (Needs HF_TOKEN)

```bash
# Set token and enter development shell
export HF_TOKEN=your_huggingface_token
nix develop
```

The shellHook automatically:
1. Checks if models exist in `./models/`
2. If not, downloads them using HF_TOKEN
3. Reports offline readiness status

### Subsequent Runs (No Token Needed)

```bash
nix develop
python -m transcribe audio.wav
```

### To Re-Download Fresh Models

```bash
rm -rf ./models/whisper/
nix develop
export HF_TOKEN=your_token
```

### To Force Online Mode (Always Download)

If local models are corrupted or outdated, simply delete them:
```bash
rm -rf ./models/
export HF_TOKEN=your_token
nix develop
```

## Model Storage

### Directory Structure

```
models/
├── whisper/
│   └── openai_whisper-large-v3/
│       ├── config.json
│       ├── tokenizer.json
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       └── ... (model files)
└── pyannote/
    ├── segmentation-3.0/
    │   └── ... (segmentation model files)
    └── speaker-diarization-3.1/
        └── ... (diarization model files)
```

### Model Size Estimates

- Whisper large-v3: ~3GB
- Pyannote diarization: ~1GB
- **Total**: ~4GB

## CLI Usage

```bash
# Basic usage
python -m transcribe audio.wav

# Custom output file
python -m transcribe audio.wav -o output.txt

# Show help
python -m transcribe --help
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `large-v3` | Whisper model (runtime name) |
| `--fallback-model` | `medium` | Fallback model |
| `--language` | `ar` | Language code |
| `--chunk-duration` | `1800` | Chunk size in seconds |
| `--output`, `-o` | auto | Output file path |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | First time only | HuggingFace token for model download |

## Known Issues & Solutions

### Model Not Found Errors

If you see `Invalid model size` errors, ensure you're using the correct model names in `models.py`:
- Use short names: `large-v3`, `medium`, `tiny`, etc.
- NOT full repo IDs like `whisper-large-v3`

### Download Failures

If downloads fail with 404 errors:
- Check that `download.py` uses full repo IDs: `openai/whisper-large-v3`
- Check that `models.py` uses short names: `large-v3`

### Offline Mode Not Working

If offline mode fails even with models downloaded:
1. Check model files exist: `ls -la models/whisper/*/`
2. Ensure all required files are present (config.json, tokenizer.json, model files)
3. Try re-downloading: `rm -rf ./models/ && nix develop`

## Testing Transcription

```bash
# Download models (first time)
export HF_TOKEN=your_token
nix develop

# Run transcription
python -m transcribe ~/Downloads/mentorship_session.mp3
```

## Python API

```python
from transcribe import (
    get_device,
    load_whisper_with_fallback,
    load_diarization_pipeline,
    transcribe_audio,
    run_diarization,
    merge_transcript_and_diarization,
    format_transcript,
    save_transcript,
)

device = get_device()
model = load_whisper_with_fallback(device)
pipeline = load_diarization_pipeline()

# Transcribe
segments, info = transcribe_audio(model, "audio.wav")

# Diarize
diarization = run_diarization(pipeline, "audio.wav")

# Merge
merged = merge_transcript_and_diarization(segments, diarization)

# Output
transcript = format_transcript(merged)
save_transcript(transcript, "output.txt")
```

## Development Commands

```bash
# Enter development shell
nix develop

# Flatten flake (update lock)
nix flake update

# Build without entering
nix build

# Run tests (if any exist)
nix develop --command pytest
```

## History

This project started as a single `transcribe.py` script and was refactored into a modular package structure:

1. Initial implementation as single file
2. Modular refactoring into `transcribe/` package
3. Added offline model support
4. Several iterations to fix model naming issues (faster-whisper uses short names)
5. Current: Standard Whisper model (`large-v3`) for compatibility