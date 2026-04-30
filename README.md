# Arabic-English Audio Transcription with Speaker Diarization

## Overview

This project provides tools for transcribing mixed Arabic-English audio with speaker diarization. It uses:

- **Whisper** (`large-v3`) - for speech-to-text transcription
- **pyannote.audio** (`speaker-diarization-3.1`) - for speaker diarization

## Features

- Mixed Arabic-English audio transcription
- Speaker diarization (identifies who spoke when)
- Offline mode (models cached locally after first download)
- Long audio handling (automatic chunking)

## Requirements

- Python 3.11+
- HuggingFace account (for first-time model download only)
- ffmpeg (for audio processing)

## Installation

### Option 1: Using Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Darth-218/transcribe.git
cd transcribe

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Nix

```bash
# Enter development shell (auto-installs dependencies)
nix develop
```

## First-Time Setup

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token
```

Get your token from https://huggingface.co/settings/tokens

## Running Transcription

### Basic Usage

```bash
python -m transcribe audio.wav
python -m transcribe audio.mp3
python -m transcribe /path/to/mentorship_session.mp3
```

### Output

The script outputs:
1. **Console**: Formatted transcript with speaker labels
2. **File**: Same name as input with `_transcript.txt` suffix

Example:
```
SPEAKER_00: مرحباً بك في جلسة الإرشاد
SPEAKER_01: Thank you for joining us today
SPEAKER_00: كيف يمكنني مساعدتك
```

### Command-Line Options

```
python -m transcribe audio.wav [options]

Options:
  --model               Whisper model (default: large-v3)
  --fallback-model      Fallback model (default: medium)
  --language            Language code (default: ar)
  --chunk-duration      Chunk duration in seconds (default: 1800)
  --output, -o          Output file path
  --help                Show help message
```

## Offline Mode

After initial setup, no internet or HF_TOKEN required:

```bash
python -m transcribe audio.wav
```

To re-download fresh models:

```bash
rm -rf models/
export HF_TOKEN=your_token
python -m transcribe audio.wav  # Will re-download
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

transcript_segments, info = transcribe_audio(model, "audio.wav")
diarization_segments = run_diarization(pipeline, "audio.wav")
merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
transcript = format_transcript(merged)
save_transcript(transcript, "output.txt")
```

## Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest -m "not integration"

# Integration tests only (requires models)
pytest -m integration
```

## Documentation

For detailed documentation, see the `docs/` folder:

| Document | Description |
|----------|-------------|
| [docs/usage.md](docs/usage.md) | CLI usage and examples |
| [docs/api.md](docs/api.md) | Python API reference |
| [docs/development.md](docs/development.md) | Development internals |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | First time only | HuggingFace token for model download |

## License

See [LICENSE](LICENSE)