# Speech-to-Text Corpus Project

## 1. Project Overview

This project focuses on preparing and processing an audio corpus for speech-to-text applications.

The pipeline includes:
- Audio collection and organization.
- Audio format conversion.
- Sample-rate standardization.
- Audio cleaning and preprocessing.
- Speech segmentation.
- Segment timestamp management.
- Dataset metadata generation and validation.

**Important segmentation requirement:** Audio must be segmented without cutting words in the middle of phonemes. Segmentation should prioritize natural speech boundaries and preserve linguistic integrity.

---

## 2. Prerequisites

Install the following before starting:

| Software | Purpose |
|---|---|
| Python 3.13 | Main programming language |
| FFmpeg 9.0.2 | Audio conversion and processing |
| VS Code | Development environment |
| Git | Version control (recommended) |

### Verify installations

```powershell
python --version
ffmpeg -version
ffprobe -version
git --version
```

---

## 3. Project Environment Setup

Open PowerShell in the project directory:

```powershell
cd "E:\Speech to text\speech-corpus"
```

### Create virtual environment

```powershell
py -3.13 -m venv .venv
```

### Activate environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Your terminal should display:

```text
(.venv) PS E:\Speech to text\speech-corpus>
```

---

## 4. Install Python Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install all project dependencies using:

```powershell
python -m pip install -r requirements.txt
```

### requirements.txt

Create a file named `requirements.txt` in the project root:

```text
numpy
pandas
matplotlib
tqdm
soundfile
librosa
scipy
pydub
```

### Verify dependencies

```powershell
python -c "import numpy, pandas, matplotlib, tqdm, soundfile, librosa, scipy, pydub; print('All packages imported successfully!')"
```

---

## 5. FFmpeg Configuration

FFmpeg is installed separately from Python packages.

Verify that Python can locate it:

```powershell
python -c "import shutil; print(shutil.which('ffmpeg'))"
```

Expected output: the full path to `ffmpeg.exe`.

Useful commands:

```powershell
ffmpeg -version
ffprobe -version
```

FFmpeg will be used for:
- Audio format conversion.
- Resampling.
- Audio extraction.
- Audio segmentation.
- Audio encoding.

---

## 6. Recommended Project Structure

```text
speech-corpus/
│
├── .venv/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── segments/
│
├── metadata/
│   ├── metadata.csv
│   └── segment_metadata.csv
│
├── notebooks/
│
├── scripts/
│   ├── audio_inspection.py
│   ├── audio_preprocessing.py
│   ├── segmentation.py
│   └── validation.py
│
├── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 7. Audio Processing Principles

### Preserve original recordings
Never overwrite or modify the original raw audio files.

### Standardize audio
Ensure processed audio follows the project's agreed:
- Sample rate.
- Channel configuration.
- Audio format.
- Bit depth where applicable.

### Speech segmentation

Segmentation must:
- Respect natural speech boundaries.
- Avoid cutting words in the middle of phonemes.
- Preserve complete words whenever possible.
- Maintain accurate start and end timestamps.
- Avoid excessively short or long segments.
- Allow manual review of uncertain boundaries.

### Metadata

Each segment should have traceable information, including:
- Original audio filename.
- Segment filename.
- Start timestamp.
- End timestamp.
- Duration.
- Transcription, when available.
- Validation status.

---

## 8. Useful Commands

### Check installed packages

```powershell
pip list
```

### Save installed dependencies

```powershell
pip freeze > requirements-lock.txt
```

### Install dependencies again

```powershell
pip install -r requirements.txt
```

### Deactivate virtual environment

```powershell
deactivate
```

### Reactivate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 9. Git Configuration

Create a `.gitignore` file:

```text
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.env
data/raw/
data/processed/
data/segments/
outputs/
```

Do not upload large audio datasets or your virtual environment to GitHub.

---

## 10. Troubleshooting

### PowerShell blocks activation

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

### FFmpeg is not recognized

Refresh the current terminal's PATH:

```powershell
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
```

### Python package installation fails

Check Python version:

```powershell
python --version
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

If a package does not support Python 3.13, consider creating a Python 3.11 environment rather than forcing an incompatible installation.

---

## 11. Project Quality Checklist

- [ ] Original recordings preserved.
- [ ] Audio formats standardized.
- [ ] Sample rates validated.
- [ ] Segmentation boundaries reviewed.
- [ ] No words cut in the middle of phonemes.
- [ ] Timestamps validated.
- [ ] Metadata generated.
- [ ] Corrupted or invalid segments identified.
- [ ] Final dataset reviewed.

**Core principle:** Dataset quality and accurate speech boundaries are more important than maximizing the number of generated segments.