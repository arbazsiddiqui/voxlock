import io
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

from wesep.dataset.FRAM_RIR import single_channel as RIR_sim
from wesep.dataset.lmdb_data import LmdbData
from wesep.dataset.timeline import sample_num_speakers, timeline_generator, parse_timeline, parse_overlap_ratio

AUDIO_FORMAT_SETS = {"flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"}


def url_opener(data):
    """Give url or local file, return file descriptor
    Inplace operation.

    Args:
        data(Iterable[str]): url or local file list

    Returns:
        Iterable[{src, stream}]
    """
    for sample in data:
        assert "src" in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample["src"]
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == "" or pr.scheme == "file":
                stream = open(url, "rb")
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f"wget -q -O - {url}"
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning("Failed to open {}".format(url))


def tar_file_and_group(data):
    """Expand a stream of open tar files into a stream of tar file contents.
    And groups the file with same prefix

    Args:
        data: Iterable[{src, stream}]

    Returns:
        Iterable[{key, mix_wav, spk1_wav, spk2_wav, ..., sample_rate}]
    """
    for sample in data:
        stream = tarfile.open(fileobj=sample["stream"], mode="r:*")

        prev_prefix = None
        example = {}
        valid = True
        num_speakers = 0

        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind(".")
            if pos <= 0:
                continue

            prefix, postfix = name[:pos], name[pos + 1:]

            # ---- new sample ----
            if prev_prefix is not None and not prefix.startswith(prev_prefix):
                example["key"] = prev_prefix
                if valid:
                    example["num_speaker"] = num_speakers
                    for k in list(example.keys()):
                        if k.startswith("wav_"):
                            example[k] = torch.cat(example[k], dim=0)
                    yield example
                example = {}
                valid = True
                num_speakers = 0

            with stream.extractfile(tarinfo) as f:
                try:
                    if postfix.startswith("spk"):
                        example[postfix] = f.read().decode("utf8").strip()
                        num_speakers += 1

                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sr = torchaudio.load(f)

                        if prefix.endswith("_spk1"):
                            example.setdefault("wav_spk1", []).append(waveform)
                            prefix = prefix[:-5]
                        elif prefix.endswith("_spk2"):
                            example.setdefault("wav_spk2", []).append(waveform)
                            prefix = prefix[:-5]
                        else:
                            example.setdefault("wav_mix", []).append(waveform)
                            example["sample_rate"] = sr

                except Exception as ex:
                    valid = False
                    logging.warning(f"Failed to parse {name}: {ex}")

            prev_prefix = prefix

        if prev_prefix is not None:
            example["key"] = prev_prefix
            example["num_speaker"] = num_speakers
            for k in list(example.keys()):
                if k.startswith("wav_"):
                    example[k] = torch.cat(example[k], dim=0)
            yield example

        stream.close()
        sample["stream"].close()


def tar_file_and_group_single_spk(data):
    """Expand a stream of open tar files into a stream of tar file contents.
    And groups the file with same prefix

    Args:
        data: Iterable[{src, stream}]

    Returns:
        Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert "stream" in sample
        stream = tarfile.open(fileobj=sample["stream"],
                              mode="r|*")  # Only support pytorch version <2.0
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind(".")
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example["key"] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ["spk"]:
                        example[postfix] = (
                            file_obj.read().decode("utf8").strip())
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example["wav"] = waveform
                        example["sample_rate"] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning("error to parse {}".format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example["key"] = prev_prefix
            yield example
        stream.close()
        if "process" in sample:
            sample["process"].communicate()
        sample["stream"].close()


def parse_raw(data):
    """Parse samples.jsonl line into wav tensors.

    Args:
        data: Iterable[dict], each item like:
          {
            "src": "<json string>",
            "rank": ...,
            "world_size": ...,
            ...
          }

    Yields:
        dict with fields aligned to shards:
          {
            key,
            num_speaker,
            spk1, spk2, ...
            wav_mix,
            wav_spk1, wav_spk2, ...
            sample_rate
          }
    """
    for sample in data:
        assert "src" in sample
        json_line = sample["src"]

        try:
            obj = json.loads(json_line)
        except Exception as ex:
            logging.warning(f"Bad json line: {json_line}")
            continue

        # --- required fields ---
        key = obj["key"]
        spk_ids = obj["spk"]  # e.g. ["412", "2836"]
        mix_dict = obj["mix"]  # {"default": [mix.wav]}
        src_dict = obj["src"]  # {"412": [...], "2836": [...]}

        num_speaker = len(spk_ids)

        # --- load mix ---
        # current convention: use default channel, first path
        mix_paths = mix_dict.get("default", [])
        if len(mix_paths) == 0:
            logging.warning(f"No mix path for sample {key}")
            continue

        #########################
        ### Handle with multi-channel mix, should notice the target is still choiced with the default first file
        wav_list = []
        sample_rate = None
        min_len = None

        for ch_idx, mix_path in enumerate(mix_paths):
            try:
                wav_ch, sr = torchaudio.load(mix_path)  # (1, T) or (T,)
            except Exception:
                logging.warning(f"Failed to read mix wav: {mix_path}")
                wav_list = []
                break

            # normalize shape to (1, T)
            if wav_ch.dim() == 1:
                wav_ch = wav_ch.unsqueeze(0)
            elif wav_ch.dim() != 2:
                logging.warning(
                    f"Unsupported number of channels in mix wav: {mix_path}, shape={tuple(wav_ch.shape)}"
                )  # noqa
                wav_list = []
                break

            if sample_rate is None:
                sample_rate = sr
                min_len = wav_ch.size(-1)
            else:
                if sr != sample_rate:
                    logging.warning(f"Sample rate mismatch in {key}: "
                                    f"mix={sample_rate}, ch{ch_idx}={sr}")
                min_len = min(min_len, wav_ch.size(-1))

            wav_list.append(wav_ch)

        if len(wav_list) == 0:
            continue

        # length align (safe default)
        if any(w.size(-1) != min_len for w in wav_list):
            wav_list = [w[..., :min_len] for w in wav_list]

        # stack into (C, T)
        if len(wav_list) == 1:
            wav_mix = wav_list[0]
            if wav_mix.dim() == 1:
                wav_mix = wav_mix.unsqueeze(0)  # To avoid additional dimension
        else:
            wav_mix = torch.stack(wav_list, dim=0)
        #########################

        example = {
            "key": key,
            "num_speaker": num_speaker,
            "wav_mix": wav_mix,
            "sample_rate": sample_rate,
        }

        # --- load sources ---
        for i, spk_id in enumerate(spk_ids, start=1):
            if spk_id not in src_dict:
                logging.warning(
                    f"Speaker {spk_id} not in src for sample {key}")
                continue

            src_paths = src_dict[spk_id]
            if len(src_paths) == 0:
                logging.warning(
                    f"No src path for speaker {spk_id} in sample {key}")
                continue

            src_path = src_paths[0]

            try:
                wav_spk, sr = torchaudio.load(src_path)
            except Exception:
                logging.warning(f"Failed to read src wav: {src_path}")
                continue

            if sr != sample_rate:
                logging.warning(f"Sample rate mismatch in {key}: "
                                f"mix={sample_rate}, src={sr}")

            example[f"spk{i}"] = spk_id
            example[
                f"wav_spk{i}"] = wav_spk[:
                                         1, :]  # Only obtain the first channel as target
        yield example


def parse_raw_single_spk(data):
    """
    Parse raw single-speaker samples for online mix.

    Input sample schema (from samples.jsonl):
    {
      "key": "...",
      "spk": ["id10001"],
      "src": {
        "id10001": [".../00001.wav"]
      }
    }

    Yields:
    {
      "key": str,
      "spk": str,
      "wav": Tensor [1, T],
      "sample_rate": int
    }
    """

    for sample in data:
        # ---- FIX: decode samples.jsonl line if needed ----
        if "spk" not in sample:
            if "src" in sample and isinstance(sample["src"], str):
                sample = json.loads(sample["src"])
            else:
                raise ValueError(
                    f"Unexpected sample format: keys={sample.keys()}")
        # -------- sanity checks --------
        spk_list = sample.get("spk", [])
        if len(spk_list) != 1:
            raise ValueError(
                f"parse_raw_single_spk expects single speaker, "
                f"got {len(spk_list)} in sample {sample.get('key')}")

        spk = spk_list[0]

        src_map = sample.get("src", {})
        if spk not in src_map:
            raise KeyError(
                f"Speaker {spk} missing in src map for sample {sample.get('key')}"
            )

        wav_list = src_map[spk]

        # -------- explicitly forbid multi-audio (future multi-channel) --------
        if len(wav_list) != 1:
            raise NotImplementedError(
                f"Multiple audio files per speaker are not supported yet "
                f"(got {len(wav_list)}) in sample {sample.get('key')}")

        wav_path = wav_list[0]

        # -------- load audio --------
        try:
            wav_ch, sr = torchaudio.load(wav_path)  # (C, T) or (T,)
        except Exception:
            logging.warning(f"Failed to read wav: {wav_path}")
            continue

        # -------- normalize shape to [1, T] --------
        if wav_ch.dim() == 1:
            wav = wav_ch.unsqueeze(0)
        else:
            if wav_ch.size(0) != 1:
                raise NotImplementedError(
                    f"Multi-channel wav is not supported yet: "
                    f"{wav_path}, shape={tuple(wav_ch.shape)}")
            wav = wav_ch

        yield {
            "key": sample["key"],
            "spk": spk,
            "wav": wav,  # [1, T]
            "sample_rate": sr,
        }


def sample_speaker_group(data,
                         num_speakers=None,
                         shuffle_size=1000,
                         timeline_conf=None,
                         rng=random):
    """Sample and pack speakers into group when loading data,
    shuffle is not needed if this function is used
    Args:
        :param data: Iterable[{key, wavs, spks}]
        :param num_speaker:
        :param shuffle_size:
        :param timeline:
    Returns:
        Iterable[{key, spk1, wav_spk1, timeline_spk1, spk2, ..., num_speaker, overlap_ratio_2spk, sample_rate}]
    """
    assert num_speakers is not None

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                num_speaker = sample_num_speakers(num_speakers, rng)
                if timeline_conf is not None:
                    timeline, overlap_ratio = timeline_generator(
                        timeline_conf, num_speaker, rng)
                else:
                    timeline = [{
                        "speaker": i,
                        "start": 0.0,
                        "end": 1.0
                    } for i in range(num_speaker)]
                    overlap_ratio = {"overlap_ratio": 1.0}

                cur_spk = x["spk"]
                example = {
                    "key": x["key"],
                    "wav_spk1": x["wav"],
                    "spk1": x["spk"],
                    "sample_rate": x["sample_rate"],
                    "num_speaker": num_speaker,
                    "overlap_ratio_2spk": parse_overlap_ratio(overlap_ratio),
                }
                # attach timeline for spk1
                example["timeline_spk1"] = parse_timeline(
                    [t for t in timeline if t["speaker"] == 0])
                key = "mix_" + x["key"]
                interference_idx = 1
                while interference_idx < num_speaker:
                    interference = random.choice(buf)
                    while interference["spk"] == cur_spk:
                        interference = random.choice(buf)
                    key = key + "_" + interference["key"]
                    interference_idx += 1
                    # attach timeline for this slot
                    example["timeline_spk" +
                            str(interference_idx)] = parse_timeline([
                                t for t in timeline
                                if t["speaker"] == (interference_idx - 1)
                            ])
                    example["wav_spk" +
                            str(interference_idx)] = interference["wav"]
                    example["spk" +
                            str(interference_idx)] = interference["spk"]
                example["key"] = key
                yield example

            buf = []

    # The samples left over
    random.shuffle(buf)
    unique_spk = list({s["spk"] for s in buf})
    K = len(unique_spk)
    for x in buf:
        num_speaker = sample_num_speakers(num_speakers, rng)
        num_speaker = min(num_speaker, K)
        if timeline_conf is not None:
            timeline, overlap_ratio = timeline_generator(
                timeline_conf, num_speaker, rng)
        else:
            timeline = [{
                "speaker": i,
                "start": 0.0,
                "end": 1.0
            } for i in range(num_speaker)]
            overlap_ratio = {"overlap_ratio": 1.0}

        cur_spk = x["spk"]
        example = {
            "key": x["key"],
            "wav_spk1": x["wav"],
            "spk1": x["spk"],
            "sample_rate": x["sample_rate"],
            "num_speaker": num_speaker,
            "overlap_ratio_2spk": parse_overlap_ratio(overlap_ratio),
        }
        # attach timeline for spk1
        example["timeline_spk1"] = parse_timeline(
            [t for t in timeline if t["speaker"] == 0])
        key = "mix_" + x["key"]
        interference_idx = 1
        while interference_idx < num_speaker:
            interference = random.choice(buf)
            while interference["spk"] == cur_spk:
                interference = random.choice(buf)
            key = key + "_" + interference["key"]
            interference_idx += 1
            # attach timeline for this slot
            example["timeline_spk" + str(interference_idx)] = parse_timeline([
                t for t in timeline if t["speaker"] == (interference_idx - 1)
            ])
            example["wav_spk" + str(interference_idx)] = interference["wav"]
            example["spk" + str(interference_idx)] = interference["spk"]
        example["key"] = key
        yield example


def apply_timeline(data):
    """
    Apply timeline masks to each speaker waveform in example.

    Args:
        data: Iterable[example], where example contains:
              wav_spk{i}: Tensor [1, T]
              timeline_spk{i}: list of [start, end] in [0,1]
              num_speaker: int

    Yields:
        example with wav_spk{i} masked by timeline
    """
    for sample in data:
        K = sample["num_speaker"]

        for i in range(1, K + 1):
            wav_key = f"wav_spk{i}"
            tl_key = f"timeline_spk{i}"

            wav = sample[wav_key]  # [1, T]
            timeline = sample[tl_key]  # list of [s, e]

            assert wav.dim() == 2 and wav.size(0) == 1, \
                f"{wav_key} must be [1, T]"

            T = wav.size(1)
            device = wav.device

            # build mask
            mask = torch.zeros(T, device=device)

            for seg in timeline:
                s, e = seg
                # clamp just in case
                s = max(0.0, min(1.0, float(s)))
                e = max(0.0, min(1.0, float(e)))
                if e <= s:
                    continue

                start = int(round(s * T))
                end = int(round(e * T))
                start = max(0, min(T, start))
                end = max(0, min(T, end))

                if end > start:
                    mask[start:end] = 1.0

            # apply
            wav = wav * mask.unsqueeze(0)  # [1, T]
            sample[wav_key] = wav
        yield sample


def snr_mixer(data, snr_conf=None, rng=random):
    """Dynamic mixing speakers when loading data, shuffle is not needed if this function is used.

    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        snr_conf:
            range: range of target-to-interference ratio in dB (after reverb)
            gain: adjust the overall energy of mix, spk1, spk2.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
    """
    for sample in data:
        assert "num_speaker" in sample.keys()
        if snr_conf is None:
            snr_conf = {
                "range": [-5, 10],
                "gain": [-12, 0],
            }
        if "wav_spk1_reverb" in sample.keys():
            suffix = "_reverb"  # Reserved when maintian the dry wav
        else:
            suffix = ""
        num_speaker = sample["num_speaker"]
        wavs_to_mix = [sample["wav_spk1" + suffix]]
        target_energy = torch.sum(wavs_to_mix[0]**2, dim=-1, keepdim=True)
        for i in range(1, num_speaker):
            snr = rng.uniform(*snr_conf["range"])
            interference = sample[f"wav_spk{i + 1}" + suffix]
            energy = torch.sum(interference**2, dim=-1, keepdim=True)
            interference *= torch.sqrt(
                target_energy / energy) * 10**(-snr / 20)
            wavs_to_mix.append(interference)
        wavs_to_mix = torch.stack(wavs_to_mix)
        sample["wav_mix"] = torch.sum(wavs_to_mix, 0)
        # ---------- Peak normalization ----------
        max_amp = max(
            torch.abs(sample["wav_mix"]).max().item(),
            *[x.item() for x in torch.abs(wavs_to_mix).max(dim=-1)[0]],
        )
        if max_amp > 0:
            peak_scale = 1.0 / max_amp
        else:
            peak_scale = 1.0
        sample["wav_mix"] *= peak_scale
        for i in range(num_speaker):
            sample[f"wav_spk{i + 1}{suffix}"] *= peak_scale
        # ---------- Random global gain (after peak norm) ----------
        if snr_conf.get("gain", None) is not None:
            gain_db = rng.uniform(*snr_conf["gain"])  # e.g. [-12, 0]
            gain = 10**(gain_db / 20)
            sample["wav_mix"] *= gain
            for i in range(num_speaker):
                sample[f"wav_spk{i + 1}{suffix}"] *= gain
        yield sample


def shuffle(data, shuffle_size=2500):
    """Local shuffle the data

    Args:
        data: Iterable[{key, wavs, spks}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, wavs, spks}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def resample(data, resample_rate=16000):
    """Resample data.
    Inplace operation.
    Args:
        data: Iterable[{key, wavs, spks, sample_rate}]
        resample_rate: target resample rate
    Returns:
        Iterable[{key, wavs, spks, sample_rate}]
    """
    for sample in data:
        assert "sample_rate" in sample
        sample_rate = sample["sample_rate"]
        if sample_rate != resample_rate:
            all_keys = list(sample.keys())
            sample["sample_rate"] = resample_rate
            for key in all_keys:
                if "wav" in key:
                    waveform = sample[key]
                    sample[key] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=resample_rate)(waveform)
        yield sample


def get_random_chunk(data_list, chunk_len):
    """
    Args:
        data_list: list[Tensor], shapes like
                   mix, targets: [1, C, T] or [1, T]
        chunk_len: int

    Returns:
        list[Tensor], same leading dims, last dim = chunk_len
    """
    # 1. extend the dim to 2 if needed, to unify the processing of mix and target
    normed = []
    for d in data_list:
        if d.dim() == 1:
            d = d.unsqueeze(0)  # [1, T]
        normed.append(d)

    # 2. check the time length and get T
    T = normed[0].size(-1)
    assert all(d.size(-1) == T for d in normed)

    # 3. random chunk if possible
    if T >= chunk_len:
        chunk_start = random.randint(0, T - chunk_len)

        out = []
        for d in normed:
            chunk = d[..., chunk_start:chunk_start + chunk_len]

            while torch.all(chunk == 0):
                chunk_start = random.randint(0, T - chunk_len)
                chunk = d[..., chunk_start:chunk_start + chunk_len]

            out.append(chunk.clone())

        meta = {
            "start_ratio": chunk_start / T,
            "end_ratio": (chunk_start + chunk_len) / T,
            "orig_len": T,
            "chunk_len": chunk_len,
        }
        return out, meta

    # 4. padding / repeat
    repeat_factor = chunk_len // T + 1
    out = []
    for d in normed:
        d_rep = d.repeat(*([1] * (d.dim() - 1)), repeat_factor)
        out.append(d_rep[..., :chunk_len].clone())

    meta = {
        "start_ratio": 0.0,
        "end_ratio": chunk_len / T,  # > 1.0, indicating repeated
        "orig_len": T,
        "chunk_len": chunk_len,
    }

    return out, meta


def filter_len(
    data,
    min_num_seconds=1,
    max_num_seconds=1000,
):
    """Filter the utterance with very short duration and random chunk the
    utterance with very long duration.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        min_num_seconds: minimum number of seconds of wav file
        max_num_seconds: maximum number of seconds of wav file
    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert "key" in sample
        assert "sample_rate" in sample
        assert "wav" in sample
        sample_rate = sample["sample_rate"]
        wav = sample["wav"]
        min_len = min_num_seconds * sample_rate
        max_len = max_num_seconds * sample_rate
        if wav.size(-1) < min_len:
            continue
        elif wav.size(-1) > max_len:
            wav, _ = get_random_chunk(
                [wav], max_len
            )[0]  # may result in spliting an utterance, the ratio should be recorded if needed
        sample["wav"] = wav
        yield sample


def random_chunk(data, chunk_len):
    """Random chunk the data into chunk_len

    Args:
        data: Iterable[{key, wav/feat, label}]
        chunk_len: chunk length for each sample

    Returns:
        Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert "key" in sample
        wav_keys = [key for key in list(sample.keys()) if "wav" in key]
        wav_data_list = [sample[key] for key in wav_keys]
        wav_data_list, ratio = get_random_chunk(wav_data_list, chunk_len)
        sample.update(zip(wav_keys, wav_data_list))
        sample["chunk_ratio"] = ratio
        yield sample


def fix_chunk(data, chunk_len):
    """Random chunk the data into chunk_len

    Args:
        data: Iterable[{key, wav/feat, label}]
        chunk_len: chunk length for each sample

    Returns:
        Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert "key" in sample
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith("wav"):
                sample[key] = sample[key][..., :chunk_len]
        yield sample


def add_noise(
    data,
    noise_lmdb_file,
    noise_prob: float = 0.0,
    noise_db_low: int = -5,
    noise_db_high: int = 25,
    single_channel: bool = True,
):
    """Add noise to mixture

    Args:
        data: Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]
        noise_lmdb_file: noise LMDB data source.
        noise_db_low (int, optional): SNR lower bound. Defaults to -5.
        noise_db_high (int, optional): SNR upper bound. Defaults to 25.
        single_channel (bool, optional): Whether to force the noise file to be single channel.  # noqa
                                         Defaults to True.

    Returns:
        Iterable[{key, wav_mix, wav_spk1, wav_spk2, ..., spk1, spk2, ..., noise, snr}]  # noqa
    """
    noise_source = LmdbData(noise_lmdb_file)
    for sample in data:
        if noise_prob > random.random():
            assert "sample_rate" in sample.keys()
            tgt_fs = sample["sample_rate"]
            speech = sample["wav_mix"].numpy()  # [1, nsamples]
            nsamples = speech.shape[1]
            power = (speech**2).mean()
            noise_key, noise_data = noise_source.random_one()
            if noise_key.startswith(
                    "speech"):  # using interference speech as additive noise
                snr_range = [10, 30]
            else:
                snr_range = [noise_db_low, noise_db_high]
            noise_db = np.random.uniform(snr_range[0], snr_range[1])
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
                        [(offset, nsamples_ - f.frames - offset), (0, 0)],
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples_)
                    f.seek(offset)
                    # noise: (Time, Nmic)
                    noise = f.read(nsamples_, dtype=np.float64, always_2d=True)
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
                noise = librosa.resample(noise,
                                         orig_sr=fs,
                                         target_sr=tgt_fs,
                                         res_type="kaiser_fast")
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
            sample["wav_mix"] = torch.from_numpy(speech)
            sample["noise"] = torch.from_numpy(scaled_noise)
            sample["snr"] = noise_db
        yield sample


def add_reverb(data, reverb_prob=0, reverb_conf=None, rng=random):
    """
    Args:
        data: Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Returns:
        Iterable[{key, wav_spk1, wav_spk2, ..., spk1, spk2, ...}]

    Note: This function is implemented with reference to
    Fast Random Appoximation of Multi-channel Room Impulse Response (FRAM-RIR)
    https://arxiv.org/pdf/2304.08052
        This function is only used when online_mixing.
    """
    if reverb_prob == 0:
        yield from data
        return

    if reverb_conf is None:
        # set the default simulation configuration
        reverb_conf = {
            "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
            "rt60": [0.1, 0.7],
            "mic_dist": [0.2, 5.0],
        }

    for sample in data:
        apply = rng.random() < reverb_prob
        if not apply:
            yield sample
            continue

        assert "num_speaker" in sample.keys()
        assert "sample_rate" in sample.keys()

        reverb_conf["num_src"] = sample["num_speaker"]
        reverb_conf["sr"] = sample["sample_rate"]

        rirs, _ = RIR_sim(reverb_conf)
        rirs = rirs[0]  # [num_speaker, rir_len]

        # Apply room impulse response to each speaker signal.
        # This simulates the acoustic propagation from speaker to microphone.
        # The reverberant signals are used as ground truth for separation.
        for i in range(sample["num_speaker"]):
            audio = sample[f"wav_spk{i+1}"].numpy()
            rir = rirs[i:i + 1]
            out = signal.convolve(audio, rir, mode="full")[:, :audio.shape[1]]
            sample[f"wav_spk{i+1}"] = torch.from_numpy(out)

        yield sample
