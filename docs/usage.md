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
| `--model` | - | `large-v3` | Whisper model size |
| `--fallback-model` | - | `medium` | Fallback model if primary fails |
| `--language` | - | `ar` | Language code (ar=Arabic, en=English) |
| `--chunk-duration` | - | `1800` | Chunk size in seconds (30 min) |
| `--output` | `-o` | auto | Output file path |

### Examples

```bash
# Basic transcription
python -m transcribe mentorship_session.wav

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

### Console Output

The script outputs formatted transcript to stdout:

```
SPEAKER_00: مرحباً بك في جلسة الإرشاد
SPEAKER_01: Thank you for joining us today
SPEAKER_00: كيف يمكنني مساعدتك
```

### File Output

Transcript is saved to:
- Auto-named: `audio_transcript.txt` (for `audio.wav`)
- Custom: Whatever path you specify with `-o`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | First time only | HuggingFace token for model download |

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

You need to set `HF_TOKEN` for first-time model download:

```bash
export HF_TOKEN=your_token
nix develop
```

### Error: No transcription output

Check that:
1. Audio file is valid: `ffmpeg -i audio.wav`
2. Audio has speech content
3. Models are downloaded: `ls -la models/`

### Slow Transcription

Try:
- Using GPU: Ensure CUDA is available
- Smaller model: `--model medium` instead of `large-v3`
- Shorter chunks: `--chunk-duration 600`