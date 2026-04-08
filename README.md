# Arabic-English Audio Transcription with Speaker Diarization

## Overview

This project provides a Nix flake with a development shell for transcribing mixed Arabic-English audio with speaker diarization. It uses:

- **Whisper**: `oddadmix/MasriSwitch-Gemma3n-Transcriber-v1` (with fallback to `MohamedRashad/Arabic-Whisper-CodeSwitching-Edition`)
- **Diarization**: `pyannote/speaker-diarization-3.1`

## Requirements

- Nix with flakes enabled
- Python 3.11+

## First-Time Setup (One-time, requires HF_TOKEN)

On first entry, models are automatically downloaded. You need a HuggingFace token:

```bash
# Get token from https://huggingface.co/settings/tokens
# Must accept terms for:
#   - pyannote/segmentation-3.0
#   - pyannote/speaker-diarization-3.1

export HF_TOKEN=your_huggingface_token_here
nix develop
```

The shell will automatically download models to `./models/` on first run.

## Normal Usage (No token needed after setup)

```bash
nix develop
python -m transcribe audio.wav
```

### Command-line Options

```bash
# Basic usage
python -m transcribe audio.wav

# Custom output file
python -m transcribe audio.wav -o output.txt

# Show help
python -m transcribe --help
```

### Options

```
positional arguments:
  audio_file            Path to audio file (wav, mp3, etc.)

optional arguments:
  --model               Whisper model (default: oddadmix/MasriSwitch-Gemma3n-Transcriber-v1)
  --fallback-model      Fallback model (default: MohamedRashad/Arabic-Whisper-CodeSwitching-Edition)
  --language            Language code (default: ar for Arabic)
  --chunk-duration      Chunk duration in seconds for long audio (default: 1800 = 30 min)
  --output, -o         Output file path
```

## Offline Mode

After initial setup, no internet or HF_TOKEN required:

```bash
# Without HF_TOKEN
nix develop

# Works offline using cached models
python -m transcribe audio.wav
```

To re-download fresh models:
```bash
rm -rf models/
export HF_TOKEN=your_token
nix develop  # Will re-download
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
    get_output_path,
)

device = get_device()
model = load_whisper_with_fallback(device)
pipeline = load_diarization_pipeline()  # Works offline if models cached

# Transcribe
transcript_segments, info = transcribe_audio(model, "audio.wav")

# Run diarization
diarization_segments = run_diarization(pipeline, "audio.wav")

# Merge and format
merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
transcript = format_transcript(merged)

# Save
save_transcript(transcript, "output.txt")
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

## Project Structure

```
transcribe/
├── __init__.py       # Public API exports
├── __main__.py       # python -m transcribe entry point
├── cli.py            # CLI argument parsing
├── models.py         # Whisper model loading
├── diarization.py    # pyannote pipeline
├── audio.py          # audio loading and chunking
├── alignment.py       # merge transcripts with speakers
├── output.py         # formatting and file output
└── download.py       # model download utilities
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | First time only | HuggingFace token for model download |

## Long Audio Handling

Audio files longer than 30 minutes are automatically split into chunks, processed separately, then merged. You can adjust the chunk duration with `--chunk-duration`.
