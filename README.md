# Arabic-English Audio Transcription with Speaker Diarization

## Overview

This project provides a Nix flake with a development shell for transcribing mixed Arabic-English audio with speaker diarization. It uses:

- **Whisper**: `oddadmix/MasriSwitch-Gemma3n-Transcriber-v1` (with fallback to `MohamedRashad/Arabic-Whisper-CodeSwitching-Edition`)
- **Diarization**: `pyannote/speaker-diarization-3.1`

## Requirements

- Nix with flakes enabled
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

### Basic transcription

```bash
python transcribe.py audio.wav
python transcribe.py mentorship_session.mp3
```

### With custom output file

```bash
python transcribe.py audio.wav -o output.txt
```

### With custom model

```bash
python transcribe.py audio.wav --model oddadmix/MasriSwitch-Gemma3n-Transcriber-v1
```

### Options

```
positional arguments:
  audio_file            Path to audio file (wav, mp3, etc.)

optional arguments:
  --model               Whisper model (default: oddadmix/MasriSwitch-Gemma3n-Transcriber-v1)
  --fallback-model     Fallback model (default: MohamedRashad/Arabic-Whisper-CodeSwitching-Edition)
  --language           Language code (default: ar for Arabic)
  --chunk-duration     Chunk duration in seconds for long audio (default: 1800 = 30 min)
  --output, -o         Output file path
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
| `HF_TOKEN` | Yes | HuggingFace token for model access |

## Long Audio Handling

Audio files longer than 30 minutes are automatically split into chunks, processed separately, then merged. You can adjust the chunk duration with `--chunk-duration`.