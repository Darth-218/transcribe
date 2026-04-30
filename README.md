# Arabic-English Audio Transcription with Speaker Diarization

## Overview

This project provides tools for transcribing mixed Arabic-English audio with speaker diarization. It uses:

- **Whisper** (`medium`) - for speech-to-text transcription
- **pyannote.audio** (`speaker-diarization-3.1`) - for speaker diarization

## Features

- Mixed Arabic-English audio transcription
- Speaker diarization (identifies who spoke when)
- Transcription-only mode (no HF_TOKEN required)
- Offline mode (models cached locally after first download)
- Long audio handling (automatic chunking)

## Requirements

- Python 3.11+
- ffmpeg (for audio processing)
- HuggingFace account (for diarization mode only)

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

## Usage

### Full Pipeline (Transcription + Diarization)

```bash
export HF_TOKEN=your_token
python -m transcribe audio.wav
```

Output:
```
SPEAKER_00: مرحباً بك في جلسة الإرشاد
SPEAKER_01: Thank you for joining us today
SPEAKER_00: كيف يمكنني مساعدتك
```

### Transcription Only (No HF_TOKEN Required)

```bash
python -m transcribe audio.wav -t
```

Output:
```
[00:00 - 00:05] مرحباً بك في جلسة الإرشاد
[00:05 - 00:10] Thank you for joining us today
[00:10 - 00:15] كيف يمكنني مساعدتك
```

### Command-Line Options

```
python -m transcribe audio.wav [options]

Options:
  --model               Whisper model (default: medium)
  --fallback-model      Fallback model (default: small)
  --language            Language code (default: ar)
  --chunk-duration      Chunk duration in seconds (default: 1800)
  --output, -o          Output file path
  --transcription-only, -t   Skip diarization, show timestamps
  --help                Show help message
```

## Offline Mode

After initial setup, no internet or HF_TOKEN required:

```bash
# With diarization (needs HF_TOKEN first time)
export HF_TOKEN=your_token
python -m transcribe audio.wav

# Transcription only (never needs HF_TOKEN)
python -m transcribe audio.wav -t
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

# Transcription only
segments = list(transcribe_audio(model, "audio.wav"))
transcript = format_transcript(segments, transcription_only=True)

# Full pipeline
pipeline = load_diarization_pipeline()
transcript_segments = list(transcribe_audio(model, "audio.wav"))
diarization_segments = run_diarization(pipeline, "audio.wav")
merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
transcript = format_transcript(merged, transcription_only=False)
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
| `HF_TOKEN` | Diarization only | HuggingFace token (not needed with `-t` flag) |

## License

See [LICENSE](LICENSE)