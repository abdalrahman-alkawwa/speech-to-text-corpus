import soundfile as sf
import sys
from pathlib import Path


def inspect_audio(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate = sf.read(path)

    channels = 1 if audio.ndim == 1 else audio.shape[1]
    duration = len(audio) / sample_rate

    print("\nAudio successfully loaded!")
    print(f"File: {path.name}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Audio shape: {audio.shape}")
    print(f"Data type: {audio.dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(" Usage: python tests/test_audio.py path/to/audio.wav")
    else:
        inspect_audio(sys.argv[1])
