# Transcribe Documentation

This is the documentation for the transcribe project - Arabic-English audio transcription with speaker diarization.

## Documentation Structure

| Document | Description |
|----------|-------------|
| [README](../README.md) | Quick start, overview, basic usage |
| [docs/usage.md](usage.md) | Detailed CLI usage and examples |
| [docs/api.md](api.md) | Python API reference |
| [docs/development.md](development.md) | Development internals and architecture |

## Quick Links

### For Users

- **Quick start**: See [README](../README.md)
- **CLI usage**: See [docs/usage.md](usage.md)
- **Python API**: See [docs/api.md](api.md)

### For Developers

- **Development**: See [docs/development.md](development.md)
- **Architecture**: See [docs/development.md](development.md#project-structure)

## Project Overview

Transcribe uses:

- **Whisper** (`large-v3`) - for speech-to-text transcription
- **pyannote.audio** (`speaker-diarization-3.1`) - for speaker segmentation

Features:
- Mixed Arabic-English audio transcription
- Speaker diarization (identifies who spoke when)
- Offline mode (models cached locally)
- Long audio handling (automatic chunking)

## Requirements

- Nix with flakes enabled
- HuggingFace account (for initial model download only)

## Common Tasks

### First-time setup

```bash
export HF_TOKEN=your_huggingface_token
nix develop
```

### Running transcription

```bash
python -m transcribe audio.wav
```

### Checking offline status

```bash
nix develop
```

The shell will display "Offline ready: Yes" when models are downloaded.

## Getting Help

For issues or questions, please open an issue on GitHub.

## License

See [LICENSE](../LICENSE)