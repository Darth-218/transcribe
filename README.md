# Arabic-English Audio Transcription with Speaker Diarization

## Overview

This project provides a Nix flake with a development shell for transcribing mixed Arabic-English audio with speaker diarization. It uses:

- **Whisper** (`large-v3`) - for speech-to-text transcription
- **pyannote.audio** (`speaker-diarization-3.1`) - for speaker diarization

## Features

- Mixed Arabic-English audio transcription
- Speaker diarization (identifies who spoke when)
- Offline mode (models cached locally after first download)
- Long audio handling (automatic chunking)

## Requirements

- Nix with flakes enabled
- Python 3.11+
- HuggingFace account (for first-time model download only)

## Quick Start

### First-Time Setup

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token

# Enter development shell (models download automatically)
nix develop
```

Get your token from https://huggingface.co/settings/tokens

### Running Transcription

```bash
nix develop
python -m transcribe audio.wav
```

## Documentation

For detailed documentation, see the `docs/` folder:

| Document | Description |
|----------|-------------|
| [docs/usage.md](docs/usage.md) | CLI usage and examples |
| [docs/api.md](docs/api.md) | Python API reference |
| [docs/development.md](docs/development.md) | Development internals |

## Command-Line Options

```
python -m transcribe audio.wav

# Options:
--model               Whisper model (default: large-v3)
--fallback-model      Fallback model (default: medium)  
--language          Language code (default: ar)
--chunk-duration    Chunk duration in seconds (default: 1800)
--output, -o        Output file path
```

## Output

The script outputs:
1. **stdout**: Formatted transcript with speaker labels
2. **File**: Same name as input with `_transcript.txt` suffix

Example output:
```
SPEAKER_00: مرحباً بك في جلسة الإرشاد
SPEAKER_01: Thank you for joining us today
SPEAKER_00: كيف يمكنني مساعدتك
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | First time only | HuggingFace token for model download |

## Offline Mode

After initial setup, no internet or HF_TOKEN required:

```bash
nix develop
python -m transcribe audio.wav
```

To re-download fresh models:
```bash
rm -rf models/
export HF_TOKEN=your_token
nix develop
```

## License

See [LICENSE](LICENSE)