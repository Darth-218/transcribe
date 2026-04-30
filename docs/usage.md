# Usage Guide

## Command-Line Interface

### Basic Usage

```bash
python -m transcribe audio.wav
python -m transcribe audio.mp3
python -m transcribe /path/to/mentorship_session.mp3
```

### Options

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--model` | - | `medium` | Whisper model size |
| `--fallback-model` | - | `small` | Fallback model if primary fails |
| `--language` | - | `ar` | Language code (ar=Arabic, en=English) |
| `--chunk-duration` | - | `1800` | Chunk size in seconds (30 min) |
| `--output` | `-o` | auto | Output file path |
| `--transcription-only` | `-t` | False | Skip diarization, show timestamps |

### Examples

```bash
# Full pipeline (transcription + diarization)
python -m transcribe mentorship_session.wav

# Transcription only (no diarization, no HF_TOKEN needed)
python -m transcribe audio.wav -t

# With custom output file
python -m transcribe audio.wav -o my_transcript.txt

# English audio instead of Arabic
python -m transcribe audio.wav --language en

# Smaller chunks for memory-constrained systems
python -m transcribe audio.wav --chunk-duration 600

# Show all options
python -m transcribe --help
```

## Output

### Full Pipeline Output

Speaker labels with text:
```
SPEAKER_00: مرحباً بك في جلسة الإرشاد
SPEAKER_01: Thank you for joining us today
SPEAKER_00: كيف يمكنني مساعدتك
```

### Transcription-Only Output (`-t` flag)

Timestamps with text:
```
[00:00 - 00:05] مرحباً بك في جلسة الإرشاد
[00:05 - 00:10] Thank you for joining us today
[00:10 - 00:15] كيف يمكنني مساعدتك
```

### File Output

Transcript is saved to:
- Auto-named: `audio_transcript.txt` (for `audio.wav`)
- Custom: Whatever path you specify with `-o`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Diarization only | HuggingFace token (not needed with `-t` flag) |

### Setting the Token

```bash
# Temporary (current session)
export HF_TOKEN=hf_xxxxx

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export HF_TOKEN=hf_xxxxx' >> ~/.bashrc
```

## Offline Mode

### First-Time Setup

```bash
# 1. Set your HuggingFace token
export HF_TOKEN=your_token

# 2. Enter development shell (models download automatically)
nix develop
```

### Subsequent Runs

```bash
# No token needed after first download
nix develop
python -m transcribe audio.wav
```

### Re-downloading Models

```bash
# Delete local models
rm -rf ./models/

# Re-download with token
export HF_TOKEN=your_token
nix develop
```

## Audio Format Support

Supported formats (via ffmpeg):
- WAV
- MP3
- FLAC
- OGG
- M4A
- WebM

## Troubleshooting

### Error: HF_TOKEN required

You need to set `HF_TOKEN` for diarization mode:

```bash
export HF_TOKEN=your_token
nix develop
```

Or use transcription-only mode:

```bash
python -m transcribe audio.wav -t
```

### Error: No transcription output

Check that:
1. Audio file is valid: `ffmpeg -i audio.wav`
2. Audio has speech content
3. Models are downloaded: `ls -la models/`

### Slow Transcription

Try:
- Using GPU: Ensure CUDA is available
- Smaller model: `--model small` instead of `medium`
- Shorter chunks: `--chunk-duration 600`
