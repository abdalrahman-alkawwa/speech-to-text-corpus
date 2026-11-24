"""Unit tests for the current local Cohere Engine B integration."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from src import transcriber as tr


class FakeInputs(dict):
    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self


class FakeProcessor:
    def __init__(self, text="recognized words"):
        self.text = text
        self.audio = None
        self.kwargs = None

    def __call__(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return FakeInputs(input_features=torch.zeros((1, 1, 8)))

    def decode(self, outputs, **kwargs):
        return self.text


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32
    generation_config = SimpleNamespace(eos_token_id=2)

    def __init__(self, sequence):
        self.sequence = sequence
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.sequence


def _wav(path: Path, sample_rate=16_000):
    sf.write(path, np.full(sample_rate, 0.1, dtype=np.float32), sample_rate, subtype="PCM_16")
    return path


def test_cohere_receives_full_float32_16khz_arabic_audio(tmp_path):
    processor = FakeProcessor("ده اختبار")
    model = FakeModel(torch.tensor([[99, 101, 102, 2]]))

    result = tr.transcribe_cohere(
        processor,
        model,
        _wav(tmp_path / "clip.wav"),
    )

    assert processor.audio.dtype == np.float32
    assert processor.audio.shape == (16_000,)
    assert processor.kwargs["sampling_rate"] == 16_000
    assert processor.kwargs["language"] == "ar"
    assert model.generate_kwargs["max_new_tokens"] == tr.COHERE_MAX_NEW_TOKENS
    assert result["asr_b_text"] == "ده اختبار"
    assert result["asr_b_generated_tokens"] == 3
    assert result["asr_b_hit_token_limit"] is False
    assert result["asr_b_quality_flags"] == []


def test_cohere_rejects_wrong_sample_rate(tmp_path):
    processor = FakeProcessor()
    model = FakeModel(torch.tensor([[99, 2]]))

    try:
        tr.transcribe_cohere(processor, model, _wav(tmp_path / "clip.wav", 22_050))
    except ValueError as exc:
        assert "Expected 16000 Hz audio" in str(exc)
    else:
        raise AssertionError("non-16-kHz input should be rejected")

    assert processor.audio is None


def test_token_limit_and_repetition_are_recorded_without_changing_text(tmp_path):
    transcript = " ".join(["نفسي"] * 12)
    processor = FakeProcessor(transcript)
    sequence = torch.tensor([[99] + [101] * tr.COHERE_MAX_NEW_TOKENS])
    model = FakeModel(sequence)

    result = tr.transcribe_cohere(processor, model, _wav(tmp_path / "clip.wav"))

    assert result["asr_b_text"] == transcript
    assert result["asr_b_generated_tokens"] == tr.COHERE_MAX_NEW_TOKENS
    assert result["asr_b_hit_token_limit"] is True
    assert "token_limit_reached" in result["asr_b_quality_flags"]
    assert "repetition_loop" in result["asr_b_quality_flags"]


def test_character_loop_is_flagged():
    assert "character_loop" in tr._transcript_quality_flags("ا" * 32)
    assert "character_loop" in tr._transcript_quality_flags("ـ" * 32)
    assert "character_loop" not in tr._transcript_quality_flags("في البداية من الصفر")
