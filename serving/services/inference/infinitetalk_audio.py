import os
import subprocess
from typing import Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from src.audio_analysis.wav2vec2 import Wav2Vec2Model


def custom_init(device: str, wav2vec_dir: str) -> Tuple[Wav2Vec2FeatureExtractor, torch.nn.Module]:
    import torch
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def loudness_norm(audio_array: np.ndarray, sr: int = 16000, lufs: float = -23) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def extract_audio_from_video(filename: str, sample_rate: int, tmp_dir: str) -> np.ndarray:
    os.makedirs(tmp_dir, exist_ok=True)
    raw_audio_path = os.path.join(tmp_dir, os.path.basename(filename).split(".")[0] + ".wav")
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, _ = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sample_rate)
    try:
        os.remove(raw_audio_path)
    except OSError:
        pass
    return human_speech_array


def audio_prepare_single(audio_path: str, sample_rate: int = 16000, tmp_dir: str = "/tmp/infinitetalk_audio") -> np.ndarray:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return extract_audio_from_video(audio_path, sample_rate, tmp_dir)
    human_speech_array, _ = librosa.load(audio_path, sr=sample_rate)
    return loudness_norm(human_speech_array, sample_rate)


def get_embedding(
    speech_array: np.ndarray,
    wav2vec_feature_extractor: Wav2Vec2FeatureExtractor,
    audio_encoder: torch.nn.Module,
    sr: int = 16000,
    device: str = "cpu",
) -> torch.Tensor:
    import torch
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # assume fps 25

    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device).unsqueeze(0)

    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        raise RuntimeError("Failed to extract audio embedding")

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d").cpu().detach()
    return audio_emb


def save_wav_16k(audio_array: np.ndarray, out_path: str, sr: int = 16000) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio_array, sr)
    return out_path

