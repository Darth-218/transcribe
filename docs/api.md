# API Reference

## Installation

```bash
nix develop
```

## Basic Usage

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

# Initialize
device = get_device()
model = load_whisper_with_fallback(device)
pipeline = load_diarization_pipeline()

# Transcribe audio
transcript_segments, info = transcribe_audio(model, "audio.wav")

# Run speaker diarization
diarization_segments = run_diarization(pipeline, "audio.wav")

# Merge results
merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)

# Format and save
transcript = format_transcript(merged)
save_transcript(transcript, "output.txt")
```

## Module: transcribe.models

### get_device()

Determine which device to use for inference.

```python
device = get_device()  # Returns "cuda" or "cpu"
```

### load_whisper_model(model_name, device, models_dir=None, hf_token=None)

Load a Whisper model.

```python
model = load_whisper_model("large-v3", "cpu")
```

### load_whisper_with_fallback(device, primary="large-v3", fallback="medium", models_dir="./models", hf_token=None)

Load Whisper model with automatic fallback.

```python
model = load_whisper_with_fallback("cpu")
```

**Constants:**
- `DEFAULT_MODEL = "large-v3"`
- `FALLBACK_MODEL = "medium"`
- `DEFAULT_MODELS_DIR = "./models"`

## Module: transcribe.diarization

### load_diarization_pipeline(hf_token=None, model_name="pyannote/speaker-diarization-3.1", models_dir="./models")

Load the speaker diarization pipeline.

```python
pipeline = load_diarization_pipeline()
# Or with HuggingFace token for initial download
pipeline = load_diarization_pipeline(hf_token="hf_xxxxx")
```

### run_diarization(pipeline, audio_path)

Run speaker diarization on an audio file.

```python
segments = run_diarization(pipeline, "audio.wav")
# Returns: [{"start": 0.0, "end": 5.5, "speaker": "SPEAKER_00"}, ...]
```

**Constants:**
- `DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"`

## Module: transcribe.audio

### get_audio_duration(audio_path)

Get the duration of an audio file in seconds.

```python
duration = get_audio_duration("audio.wav")  # Returns float
```

### split_audio_chunks(audio_path, chunk_duration=1800)

Split audio into chunks for long files.

```python
chunks = split_audio_chunks("audio.wav", chunk_duration=1800)
# Returns: [(start_time, end_time, audio_path), ...]
```

### process_chunk(model, pipeline, audio_path, start_time, end_time)

Process a single chunk of audio.

```python
result = process_chunk(model, pipeline, "audio.wav", 0.0, 1800.0)
```

### transcribe_audio(model, audio_path, language="ar")

Transcribe audio file using Whisper.

```python
segments, info = transcribe_audio(model, "audio.wav")
# segments: [{"start": 0.0, "end": 5.5, "text": "..."}, ...]
# info: Whisper transcription info object
```

## Module: transcribe.alignment

### calculate_overlap(seg_start, seg_end, dia_start, dia_end)

Calculate overlap duration between two time ranges.

```python
overlap = calculate_overlap(0.0, 5.5, 3.0, 8.0)  # Returns 2.5
```

### merge_transcript_and_diarization(transcript_segments, diarization_segments)

Merge transcript with speaker labels based on temporal overlap.

```python
merged = merge_transcript_and_diarization(transcript_segments, diarization_segments)
# Returns: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.5, "text": "..."}, ...]
```

## Module: transcribe.output

### format_transcript(merged_segments)

Format merged segments as a human-readable transcript.

```python
transcript = format_transcript(merged)
# Returns: "SPEAKER_00: Hello\nSPEAKER_01: Hi there"
```

### save_transcript(transcript, output_path)

Save transcript to a file.

```python
save_transcript(transcript, "output.txt")
```

### get_output_path(audio_file, custom_output=None)

Get the output file path.

```python
output_path = get_output_path("audio.wav")  # Returns "audio_transcript.txt"
output_path = get_output_path("audio.wav", "custom.txt")  # Returns "custom.txt"
```

## Module: transcribe.download

### download_all(models_dir="./models", hf_token=None)

Download all required models for offline use.

```python
download_all("./models", "hf_xxxxx")
```

### check_models_exist(models_dir="./models")

Check if all models are downloaded.

```python
exists = check_models_exist("./models")  # Returns True/False
```

**Constants:**
- `DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"`

## Entry Points

### CLI

```bash
python -m transcribe audio.wav
```

### Module

```python
from transcribe.cli import main
main()
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (required for first download) |

### Models Directory

Default: `./models/`

```
models/
├── whisper/
│   └── openai_whisper-large-v3/
└── pyannote/
    ├── segmentation-3.0/
    └── speaker-diarization-3.1/
```