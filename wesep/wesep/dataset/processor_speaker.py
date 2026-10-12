# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

import random
import logging

import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from scipy import signal

from wesep.dataset.FRAM_RIR import single_channel as RIR_sim
from wesep.utils.file_utils import load_json


def _build_lookup_key(sample, spk_slot, key_field):
    """
    Build lookup key for speaker cue resource.

    key_field semantics:
      - "spk_id"       -> use sample[spk_slot]
      - "mix_spk_id"   -> use f"{sample['key']}::{sample[spk_slot]}"
    """
    if key_field == "spk_id":
        return sample[spk_slot]

    elif key_field == "mix_spk_id":
        mix_key = sample.get("key", None)
        if mix_key is None:
            raise KeyError("sample missing 'key' for mix_spk_id cue")
        return f"{mix_key}::{sample[spk_slot]}"

    else:
        raise ValueError(f"Unsupported key_field for speaker cue: {key_field}")


# module-level cache (per worker process)
_SPK_RESOURCE_CACHE = {}


def _get_spk_resource(resource_path):
    """
    Lazy-load and cache speaker cue resources.

    Cache is keyed by resource_path to avoid train/val or
    multi-dataset cross-contamination.
    """
    if resource_path not in _SPK_RESOURCE_CACHE:
        _SPK_RESOURCE_CACHE[resource_path] = load_json(resource_path)
    return _SPK_RESOURCE_CACHE[resource_path]


def sample_random_speaker_cue(
    data,
    resource_path,
    key_field,
    scope="speaker",
    required=True,
):
    if scope not in ("speaker", "utterance"):
        raise ValueError(f"Unsupported scope: {scope}")

    spk_resource = _get_spk_resource(resource_path)

    for sample in data:
        spk_slots = [k for k in sample.keys() if k.startswith("spk")]

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        # choose which slots to process
        if scope == "utterance":  # Reserved
            spk_slots = [spk_slots[0]]

        for slot in spk_slots:
            spk_id = sample[slot]
            lookup_key = _build_lookup_key(sample, slot, key_field)

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(f"speaker cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(f"empty speaker cue set: {lookup_key}")
                continue

            enroll_item = random.choice(items)
            wav_path = enroll_item["path"]

            try:
                enrollment, sr = sf.read(wav_path)
            except Exception as e:
                logging.warning(
                    f"Failed to read enrollment wav: {wav_path}, err={e}")
                if required:
                    raise
                continue

            if enrollment.ndim == 1:
                enrollment = np.expand_dims(enrollment, axis=0)

            sample[f"audio_{slot}"] = enrollment

        # utterance scope → share to all slots
        if scope == "utterance":
            emb = sample[f"audio_{spk_slots[0]}"]
            for slot in [k for k in sample.keys() if k.startswith("spk")]:
                sample[f"audio_{slot}"] = emb

        yield sample


def sample_fixed_speaker_cue(
    data,
    resource_path,
    key_field,
    scope="speaker",
    required=True,
):
    if scope not in ("speaker", "utterance"):
        raise ValueError(f"Unsupported scope: {scope}")

    spk_resource = _get_spk_resource(resource_path)

    for sample in data:
        spk_slots = [k for k in sample.keys() if k.startswith("spk")]

        if not spk_slots:
            if required:
                raise KeyError("sample has no speaker slots (spk1, spk2, ...)")
            yield sample
            continue

        if scope == "utterance":
            spk_slots = [spk_slots[0]]

        for slot in spk_slots:
            lookup_key = _build_lookup_key(sample, slot, key_field)

            if lookup_key not in spk_resource:
                if required:
                    raise KeyError(
                        f"fixed speaker cue not found: {lookup_key}")
                continue

            items = spk_resource[lookup_key]
            if not items:
                if required:
                    raise RuntimeError(
                        f"empty fixed speaker cue: {lookup_key}")
                continue

            enroll_item = items[0]
            wav_path = enroll_item["path"]

            try:
                enrollment, sr = sf.read(wav_path)
            except Exception as e:
                logging.warning(
                    f"Failed to read fixed enrollment wav: {wav_path}, err={e}"
                )
                if required:
                    raise
                continue

            if enrollment.ndim == 1:
                enrollment = np.expand_dims(enrollment, axis=0)

            sample[f"audio_{slot}"] = enrollment

        if scope == "utterance":
            emb = sample[f"audio_{spk_slots[0]}"]
            for slot in [k for k in sample.keys() if k.startswith("spk")]:
                sample[f"audio_{slot}"] = emb

        yield sample


simu_config = {
    "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
    "rt60": [0.1, 0.7],
    "sr": 16000,
    "mic_dist": [0.2, 5.0],
    "num_src": 1,
}


def add_reverb_on_enroll(data, reverb_enroll_prob=0):
    """
    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Returns:
        Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        assert "sample_rate" in sample.keys()
        for i in range(sample["num_speaker"]):
            simu_config["sr"] = sample["sample_rate"]
            simu_config["num_src"] = 1
            rirs, _ = RIR_sim(simu_config)  # [n_mic, nsource, nsamples]
            rirs = rirs[0]  # [nsource, nsamples]
            if reverb_enroll_prob > random.random():
                # [1, audio_len], currently only support single-channel audio
                audio = sample[f"audio_spk{i+1}"]
                # rir = rirs[i : i + 1, :]  # [1, nsamples]
                rir = rirs
                rir_audio = signal.convolve(
                    audio, rir,
                    mode="full")[:, :audio.shape[1]]  # [1, audio_len]

                max_scale = np.max(np.abs(rir_audio))
                out_audio = rir_audio / max_scale * 0.9
                # Note: Here, we do not replace the dry audio with the reverberant audio,  # noqa
                # which means we hope the model to perform dereverberation and
                # TSE simultaneously.
                sample[f"audio_spk{i+1}"] = out_audio

        yield sample


def add_noise_on_enroll(
    data,
    noise_lmdb_file,
    noise_enroll_prob: float = 0.0,
    noise_db_low: int = 0,
    noise_db_high: int = 25,
    single_channel: bool = True,
):
    """Add noise to mixture

    Args:
        data: Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        noise_lmdb_file: noise LMDB data source.
        noise_db_low (int, optional): SNR lower bound. Defaults to 0.
        noise_db_high (int, optional): SNR upper bound. Defaults to 25.
        single_channel (bool, optional): Whether to force the noise file to be single channel.  # noqa
                                         Defaults to True.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ..., noise, snr}]  # noqa
    """

    noise_source = LmdbData(noise_lmdb_file)
    for sample in data:
        assert "sample_rate" in sample.keys()
        tgt_fs = sample["sample_rate"]
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("spk") and "label" not in key:
                if noise_enroll_prob > random.random():
                    speech = sample["audio_" + key]
                    nsamples = speech.shape[1]
                    power = (speech**2).mean()
                    noise_key, noise_data = noise_source.random_one()
                    if noise_key.startswith(
                            "speech"
                    ):  # using interference speech as additive noise
                        snr_range = [10, 30]
                    else:
                        snr_range = [noise_db_low, noise_db_high]
                    noise_db = np.random.uniform(snr_range[0], snr_range[1])
                    _, noise_data = noise_source.random_one()
                    with sf.SoundFile(io.BytesIO(noise_data)) as f:
                        fs = f.samplerate
                        if tgt_fs and fs != tgt_fs:
                            nsamples_ = int(nsamples / tgt_fs * fs) + 1
                        else:
                            nsamples_ = nsamples
                        if f.frames == nsamples_:
                            noise = f.read(dtype=np.float64, always_2d=True)
                        elif f.frames < nsamples_:
                            offset = np.random.randint(0, nsamples_ - f.frames)
                            # noise: (Time, Nmic)
                            noise = f.read(dtype=np.float64, always_2d=True)
                            # Repeat noise
                            noise = np.pad(
                                noise,
                                [
                                    (offset, nsamples_ - f.frames - offset),
                                    (0, 0),
                                ],
                                mode="wrap",
                            )
                        else:
                            offset = np.random.randint(0, f.frames - nsamples_)
                            f.seek(offset)
                            # noise: (Time, Nmic)
                            noise = f.read(nsamples_,
                                           dtype=np.float64,
                                           always_2d=True)
                            if len(noise) != nsamples_:
                                raise RuntimeError(
                                    f"Something wrong: {noise_lmdb_file}")

                    if single_channel:
                        num_ch = noise.shape[1]
                        chs = [np.random.randint(num_ch)]
                        noise = noise[:, chs]
                    # noise: (Nmic, Time)
                    noise = noise.T
                    if tgt_fs and fs != tgt_fs:
                        logging.warning(
                            f"Resampling noise to match the sampling rate ({fs} -> {tgt_fs} Hz)"  # noqa
                        )
                        noise = librosa.resample(
                            noise,
                            orig_sr=fs,
                            target_sr=tgt_fs,
                            res_type="kaiser_fast",
                        )
                        if noise.shape[1] < nsamples:
                            noise = np.pad(
                                noise,
                                [(0, 0), (0, nsamples - noise.shape[1])],
                                mode="wrap",
                            )
                        else:
                            noise = noise[:, :nsamples]
                    noise_power = (noise**2).mean()
                    scale = (10**(-noise_db / 20) * np.sqrt(power) /
                             np.sqrt(max(noise_power, 1e-10)))
                    scaled_noise = scale * noise
                    speech = speech + scaled_noise
                    sample["audio_" + key] = speech
        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """Extract fbank

    Args:
        data: Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'audio_spk1', 'audio_spk2']  # noqa

    Returns:
        Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'audio_spk1', 'audio_spk2']  # noqa
    """
    for sample in data:
        assert "sample_rate" in sample
        sample_rate = sample["sample_rate"]
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("embed"):
                waveform = torch.from_numpy(sample[key])
                waveform = waveform * (1 << 15)
                mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    sample_frequency=sample_rate,
                    window_type="hamming",
                    use_energy=False,
                )
                sample[key] = mat
        yield sample


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """Apply CMVN

    Args:
        data: Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'audio_spk1', 'audio_spk2']  # noqa

    Returns:
        Iterable['spk1', 'spk2', 'wav_mix', 'sample_rate', 'wav_spk1', 'wav_spk2', 'key', 'num_speaker', 'audio_spk1', 'audio_spk2']  # noqa
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("embed"):
                mat = sample[key]
                if norm_mean:
                    mat = mat - torch.mean(mat, dim=0)
                if norm_var:
                    mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
                mat = mat.unsqueeze(0)
                sample[key] = mat.detach().numpy()
        yield sample


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0):
    """Do spec augmentation
    Inplace operation

    Args:
        data: Iterable[{key, feat, label}]
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        prob: prob of spec_aug

    Returns
        Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            all_keys = list(sample.keys())
            for key in all_keys:
                if key.startswith("embed"):
                    y = sample[key]
                    max_frames = y.shape[1]
                    max_freq = y.shape[2]
                    # time mask
                    for i in range(num_t_mask):
                        start = random.randint(0, max_frames - 1)
                        length = random.randint(1, max_t)
                        end = min(max_frames, start + length)
                        y[:, start:end, :] = 0
                    # freq mask
                    for i in range(num_f_mask):
                        start = random.randint(0, max_freq - 1)
                        length = random.randint(1, max_f)
                        end = min(max_freq, start + length)
                        y[:, :, start:end] = 0
                    sample[key] = y
        yield sample