
import os
import site
from pathlib import Path

# Locate NVIDIA CUDA DLL directories inside the active virtual environment
nvidia_dir = Path(site.getsitepackages()[-1]) / "nvidia"

dll_handles = []

for bin_dir in nvidia_dir.glob("*/bin"):
    dll_handles.append(os.add_dll_directory(str(bin_dir)))

from faster_whisper import WhisperModel

print("Loading Whisper on NVIDIA GPU...")

model = WhisperModel(
    "tiny",
    device="cuda",
    compute_type="float16"
)

print("SUCCESS: Whisper is running on the GPU!")
