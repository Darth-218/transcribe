# Arabic-English Audio Transcription with Speaker Diarization

## Overview

This project provides a Nix flake with a development shell for transcribing mixed Arabic-English audio with speaker diarization. It uses:

- **Whisper**: `oddadmix/MasriSwitch-Gemma3n-Transcriber-v1` (with fallback to `MohamedRashad/Arabic-Whisper-CodeSwitching-Edition`)
- **Diarization**: `pyannote/speaker-diarization-3.1`

## Requirements

- Nix with flakes enabled
- Python 3.11+
- HuggingFace account with access token

## Setup

### 1. Enter the development shell

```bash
nix develop
```

### 2. Set your HuggingFace token

```bash
export HF_TOKEN=your_huggingface_token_here
```

To get a token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Accept the terms for:
   - `pyannote/segmentation-3.0`
   - `pyannote/speaker-diarization-3.1`

## Usage

### Command-line

```bash
# Using module invocation (recommended)
python -m transcribe audio.wav

# Using CLI directly
transcribe/cli.py audio.wav

# With custom output file
python -m transcribe audio.wav -o output.txt

# With custom model
python -m transcribe audio.wav --model oddadmix/MasriSwitch-Gemma3n-Transcriber-v1

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
  --output, -o          Output file path
```

### Python API

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
pipeline = load_diarization_pipeline("your_hf_token")

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
├── models.py        # Whisper model loading
├── diarization.py   # pyannote pipeline
├── audio.py         # audio loading and chunking
├── alignment.py     # merge transcripts with speakers
└── output.py        # formatting and file output
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for model access |

## Long Audio Handling

Audio files longer than 30 minutes are automatically split into chunks, processed separately, then merged. You can adjust the chunk duration with `--chunk-duration`.
