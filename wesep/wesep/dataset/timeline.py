# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: This script is to generate the timeline of speakers in online mixing.

import random


def timeline_generator(conf, num_spk=2, rng=random):

    overlap_ratio = 0
    if num_spk == 1:
        timeline = generate_single_speaker()
    elif num_spk == 2:
        timeline, overlap_ratio = generate_two_speaker(conf["two_speaker"],
                                                       rng)
    else:
        timeline, overlap_ratio = generate_multi_speaker(conf, num_spk, rng)

    timeline = apply_head_tail_silence(
        timeline,
        conf["silence"],
        rng,
    )

    return timeline, {"overlap_ratio": overlap_ratio}


def sample_num_speakers(conf, rng):
    probs = conf["distribution"]
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6

    bucket = rng.choices([1, 2, 3], probs)[0]

    if bucket == 3:
        return rng.randint(3, conf.get("max_speakers", 3))
    return bucket


# ===== timeline is always in [0, 1] =====


def generate_single_speaker():
    return [{
        "speaker": 0,
        "start": 0.0,
        "end": 1.0,
    }]


def generate_two_speaker(conf, rng):
    overlap_ratio = rng.uniform(*conf["overlap_ratio"])
    pos = rng.choices(
        list(conf["overlap_position"].keys()),
        list(conf["overlap_position"].values()),
    )[0]

    if pos == "head":
        return two_speaker_head(overlap_ratio, rng), overlap_ratio
    elif pos == "tail":
        return two_speaker_tail(overlap_ratio, rng), overlap_ratio
    else:
        return two_speaker_middle(overlap_ratio, conf, rng), overlap_ratio


def two_speaker_head(overlap_ratio, rng):
    # two speakers start together at 0
    # one ends earlier

    early_end = overlap_ratio
    late_end = rng.uniform(max(0.9, overlap_ratio), 1.0)

    if rng.random() < 0.5:
        outer, inner = 0, 1
    else:
        outer, inner = 1, 0

    return [
        {
            "speaker": outer,
            "start": 0.0,
            "end": late_end
        },
        {
            "speaker": inner,
            "start": 0.0,
            "end": early_end
        },
    ]


def two_speaker_tail(overlap_ratio, rng):
    # two speakers end together at 1
    # one starts later

    late_start = 1.0 - overlap_ratio
    early_start = rng.uniform(0.0, min(0.1, late_start))

    if rng.random() < 0.5:
        outer, inner = 0, 1
    else:
        outer, inner = 1, 0

    return [
        {
            "speaker": outer,
            "start": early_start,
            "end": 1.0
        },
        {
            "speaker": inner,
            "start": late_start,
            "end": 1.0
        },
    ]


def two_speaker_middle(overlap_ratio, conf, rng):
    modes = conf.get("middle_mode", {"crossing": 0.5, "containment": 0.5})
    mode = rng.choices(list(modes.keys()), list(modes.values()))[0]

    if mode == "containment":
        return two_speaker_containment(overlap_ratio, rng)
    else:
        return two_speaker_crossing(overlap_ratio, rng)


def two_speaker_containment(overlap_ratio, rng):
    # choose who is outer
    if rng.random() < 0.5:
        outer, inner = 0, 1
    else:
        outer, inner = 1, 0

    # outer covers a long span
    # outer_start = rng.uniform(0.0, 0.1)
    # outer_end = rng.uniform(0.9, 1.0)
    outer_start = 0.0
    outer_end = 1.0

    # inner sits inside outer with exact overlap length
    max_start = outer_end - overlap_ratio
    inner_start = rng.uniform(outer_start, max_start)
    inner_end = inner_start + overlap_ratio

    return [
        {
            "speaker": outer,
            "start": outer_start,
            "end": outer_end
        },
        {
            "speaker": inner,
            "start": inner_start,
            "end": inner_end
        },
    ]


def two_speaker_crossing(overlap_ratio, rng):
    overlap_start = rng.uniform(0.1, 1.0 - overlap_ratio - 0.1)
    overlap_end = overlap_start + overlap_ratio

    # decide who leads
    if rng.random() < 0.5:
        # spk0 leads
        a_start = 0.0
        a_end = overlap_end + rng.uniform(0.0, 1.0 - overlap_end)
        b_start = overlap_start
        b_end = overlap_start + (a_end - a_start)
    else:
        # spk1 leads
        b_start = 0.0
        b_end = overlap_end + rng.uniform(0.0, 1.0 - overlap_end)
        a_start = overlap_start
        a_end = overlap_start + (b_end - b_start)

    return [
        {
            "speaker": 0,
            "start": a_start,
            "end": min(a_end, 1.0)
        },
        {
            "speaker": 1,
            "start": b_start,
            "end": min(b_end, 1.0)
        },
    ]


def generate_multi_speaker(conf, num_spk, rng):
    timeline, overlap_ratio = generate_two_speaker(
        conf["two_speaker"],
        rng,
    )

    for spk in range(2, num_spk):
        active_ratio = rng.uniform(*conf["extra_speaker_activity"])
        start = rng.uniform(0.0, 1.0 - active_ratio)

        timeline.append({
            "speaker": spk,
            "start": start,
            "end": start + active_ratio,
        })

    return timeline, overlap_ratio


def apply_head_tail_silence(timeline, conf, rng):
    if not conf["allow"]:
        return timeline

    head = rng.uniform(*conf["head_tail_ratio"])
    tail = rng.uniform(*conf["head_tail_ratio"])

    shift = head
    valid_end = 1.0 - tail

    out = []
    for seg in timeline:
        s = seg["start"] + shift
        e = seg["end"] + shift
        if s < valid_end:
            out.append({
                "speaker": seg["speaker"],
                "start": s,
                "end": min(e, valid_end),
            })
    return out


def parse_timeline(x):
    """
    Input:
        [{'speaker': k, 'start': s, 'end': e}] or already [s, e]
    Return:
        [s, e] rounded to 2 decimals
    """
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
        s = float(x[0]["start"])
        e = float(x[0]["end"])
        return [[round(s, 3), round(e, 3)]]
    else:
        # already [s, e]
        return [[round(float(x[0]), 3), round(float(x[1]), 3)]]


def parse_overlap_ratio(x):
    """
    Input:
        {'overlap_ratio': 0.9197450214655407}  or float
    Return:
        float rounded to 4 decimals
    """
    if isinstance(x, dict):
        x = x["overlap_ratio"]
    return round(float(x), 4)


if __name__ == "__main__":
    num_speakers = {
        "distribution": [0.1, 0.7, 0.2],  # 3rd means 3+
        "max_speakers": 4,  # Upper bound of 3+
    }

    conf = {
        "two_speaker": {
            "overlap_ratio": [0.5, 1.0],
            "overlap_position": {
                "head": 0.3,
                "middle": 0.4,
                "tail": 0.3,
            },
            "middle_mode": {
                "crossing": 0.6,
                "containment": 0.4,
            },
        },
        "extra_speaker_activity": [0.1, 0.8],
        "silence": {
            "allow": False,
            "head_tail_ratio": [0.0, 0.1],
        },
    }
    num_spk = sample_num_speakers(num_speakers, random)
    timeline = timeline_generator(conf, num_spk, random)
    print(timeline)
