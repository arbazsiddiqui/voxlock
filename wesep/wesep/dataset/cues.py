# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

from wesep.dataset import processor_speaker
from wesep.utils.file_utils import load_yaml

# ---------------------------
# main cue entry
# ---------------------------


def build_cue_layer(dataset, cues_yaml, state, configs):
    """
    Build all cue pipelines declared in cues.yaml

    Args:
        dataset: base dataset pipeline
        cues_yaml: path to cues.yaml
        state: train / val
        configs: dataset configs

    Returns:
        dataset with all cue processors attached
    """
    cues_conf = load_yaml(cues_yaml)

    if "cues" in cues_conf:
        cues_conf = cues_conf["cues"]

    for cue_name, cue_conf in cues_conf.items():
        dataset = apply_single_cue(
            dataset,
            cue_name,
            cue_conf,
            state,
            configs,
        )

    return dataset


# ---------------------------
# cue dispatch
# ---------------------------


def apply_single_cue(dataset, cue_name, cue_conf, state, configs):
    if cue_name not in CUE_BUILDERS:
        raise ValueError(f"Unknown cue: {cue_name}")

    return CUE_BUILDERS[cue_name](
        dataset,
        cue_conf,
        state,
        configs,
    )


# registry
CUE_BUILDERS = {}


def register_cue(name):

    def deco(fn):
        CUE_BUILDERS[name] = fn
        return fn

    return deco


# ---------------------------
# audio cue
# ---------------------------
@register_cue("audio")
def build_speaker_cue(dataset, cue_conf, state, configs):
    """
    Stub for speaker cue
    """
    required = cue_conf.get("required", True)
    scope = cue_conf.get("scope", "speaker")  # "speaker", "utterance"

    policy_conf = cue_conf.get("policy", {})
    policy_type = policy_conf.get("type", None)  # "random", "fixed"
    key_field = policy_conf.get("key", None)
    resource_path = policy_conf.get("resource", None)

    if policy_type is None or key_field is None or resource_path is None:
        raise ValueError(f"Invalid speaker cue policy config: {policy_conf}")

    if policy_type == "random":
        dataset = dataset.apply(
            processor_speaker.sample_random_speaker_cue,
            resource_path,
            key_field=key_field,
            scope=scope,
            required=required,
        )

    elif policy_type == "fixed":
        dataset = dataset.apply(
            processor_speaker.sample_fixed_speaker_cue,
            resource_path,
            key_field=key_field,
            scope=scope,
            required=required,
        )

    else:
        raise ValueError(f"Unknown speaker cue policy type: {policy_type}")

    audio_conf = configs.get("cue_processing", {}).get("audio", {})
    if state == "train":
        reverb_enroll_prob = audio_conf.get("reverb_enroll_prob", 0)
        if reverb_enroll_prob > 0:
            dataset = dataset.apply(processor_speaker.add_reverb_on_enroll,
                                    reverb_enroll_prob)

        noise_enroll_prob = audio_conf.get("noise_enroll_prob", 0)
        noise_lmdb_file = configs.get("noise_lmdb_file", None)
        if noise_enroll_prob > 0:
            assert noise_lmdb_file is not None
            dataset = dataset.apply(
                processor_speaker.add_noise_on_enroll,
                noise_lmdb_file,
                noise_enroll_prob,
            )

    return dataset