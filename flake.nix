{
  description = "Development environment for Arabic-English audio transcription with speaker diarization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          pkgs.python311
          (pkgs.python311.withPackages (
            ps: with ps; [
              torch
              torchvision
              torchaudio
              faster-whisper
              pyannote-audio
              soundfile
              librosa
              numpy
              scipy
              huggingface-hub
            ]
          ))

          pkgs.ffmpeg
          pkgs.libsndfile
        ];

        shellHook = ''
          # Check and download models if needed
          if [ ! -d "models/whisper" ] || [ ! -d "models/pyannote" ]; then
            if [ -n "$HF_TOKEN" ]; then
              echo "=========================================="
              echo "Models not found. Downloading..."
              echo "=========================================="
              python -c "from transcribe.download import download_all; download_all('./models', '$HF_TOKEN')"
            else
              echo "=========================================="
              echo "WARNING: Models not found!"
              echo "Set HF_TOKEN and re-enter to download."
              echo "=========================================="
            fi
          fi

          echo ""
          echo "=========================================="
          echo "Arabic-English Audio Transcription Shell"
          echo "=========================================="
          echo ""
          echo "Available packages:"
          echo "  - Python: $(python --version)"
          echo "  - torch: $(python -c 'import torch; print(torch.__version__)')"
          echo "  - faster-whisper: $(python -c 'import faster_whisper; print(faster_whisper.__version__)')"
          echo "  - pyannote.audio: $(python -c 'import pyannote.audio; print(pyannote.audio.__version__ if hasattr(pyannote.audio, "__version__") else "installed")')"
          echo "  - ffmpeg: $(ffmpeg -version | head -1)"
          echo ""
          echo "Models:"
          echo "  - Location: ./models/"
          echo "  - Offline ready: $([ -d 'models/whisper' ] && [ -d 'models/pyannote' ] && echo 'Yes' || echo 'No')"
          echo ""
          echo "Usage:"
          echo "  python -m transcribe <audio_file>"
          echo ""
        '';
      };
    };
}
