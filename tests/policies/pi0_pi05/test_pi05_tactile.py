#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.utils import build_dataset_frame, dataset_to_policy_features, hw_to_dataset_features
from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors
from lerobot.policies.tactile.encoder import TactileTokenEncoder
from lerobot.processor import ProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_TACTILE


def _pi05_features(*tactile_keys: str) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    for key in tactile_keys:
        input_features[key] = PolicyFeature(type=FeatureType.TACTILE, shape=(12, 32))
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    return input_features, output_features


def test_dataset_to_policy_features_marks_observation_tactile_as_tactile():
    features = {
        "observation.tactile.right": {
            "dtype": "float32",
            "shape": [12, 32],
            "names": ["height", "width"],
        },
        "observation.tactile.right_raw": {
            "dtype": "float32",
            "shape": [130],
            "names": [f"tactile_{index}" for index in range(130)],
        },
    }

    policy_features = dataset_to_policy_features(features)

    assert policy_features["observation.tactile.right"].type is FeatureType.TACTILE
    assert policy_features["observation.tactile.right"].shape == [12, 32]
    assert policy_features["observation.tactile.right_raw"].type is FeatureType.TACTILE


def test_dataset_feature_helpers_keep_tactile_maps_out_of_observation_state():
    hw_features = {
        "joint1.pos": float,
        "joint2.pos": float,
        "observation.tactile.right": PolicyFeature(type=FeatureType.TACTILE, shape=(12, 32)),
    }

    dataset_features = hw_to_dataset_features(hw_features, prefix="observation", use_video=False)
    frame = build_dataset_frame(
        dataset_features,
        {
            "joint1.pos": 0.1,
            "joint2.pos": 0.2,
            "observation.tactile.right": torch.ones(12, 32).numpy(),
        },
        prefix="observation",
    )

    assert dataset_features["observation.state"]["names"] == ["joint1.pos", "joint2.pos"]
    assert dataset_features["observation.tactile.right"]["shape"] == (12, 32)
    assert frame["observation.state"].shape == (2,)
    assert frame["observation.tactile.right"].shape == (12, 32)


def test_pi05_config_auto_detects_right_and_dual_tactile_features():
    right_config = PI05Config(use_tactile=True)
    right_config.input_features, right_config.output_features = _pi05_features("observation.tactile.right")
    right_config.validate_features()

    assert right_config.tactile_features == ["observation.tactile.right"]
    assert right_config.normalization_mapping["TACTILE"] is NormalizationMode.MEAN_STD

    dual_config = PI05Config(use_tactile=True)
    dual_config.input_features, dual_config.output_features = _pi05_features(
        "observation.tactile.right",
        "observation.tactile.left",
    )
    dual_config.validate_features()

    assert dual_config.tactile_features == ["observation.tactile.left", "observation.tactile.right"]


def test_pi05_config_ignores_tactile_features_when_disabled():
    config = PI05Config(use_tactile=False)
    config.input_features, config.output_features = _pi05_features("observation.tactile.right")
    config.validate_features()

    assert config.tactile_features is None


def test_pi05_policy_extracts_auto_detected_left_right_tactile_tensors():
    config = PI05Config(use_tactile=True)
    config.input_features, config.output_features = _pi05_features(
        "observation.tactile.left",
        "observation.tactile.right",
    )
    config.validate_features()
    policy = object.__new__(PI05Policy)
    policy.config = config

    left = torch.full((2, 12, 32), 1.0)
    right = torch.full((2, 12, 32), 2.0)
    tactile_data = policy._extract_tactile_data(
        {
            "observation.tactile.right": right,
            "observation.tactile.left": left,
        }
    )

    assert tactile_data == [left, right]


def test_pi05_policy_uses_single_observation_tactile_key_by_default():
    config = PI05Config(use_tactile=True)
    config.input_features, config.output_features = _pi05_features(OBS_TACTILE)
    config.validate_features()
    policy = object.__new__(PI05Policy)
    policy.config = config

    tactile = torch.ones(1, 12, 32)

    assert policy._extract_tactile_data({OBS_TACTILE: tactile}) == [tactile]


def test_pi05_config_can_read_local_paligemma_tokenizer_from_env(monkeypatch):
    tokenizer_path = "/home/phl/workspace/models/paligemma-tokenizer"
    monkeypatch.setenv("LEROBOT_PALIGEMMA_TOKENIZER", tokenizer_path)

    config = PI05Config()

    assert config.paligemma_tokenizer_name == tokenizer_path


def test_pi05_preprocessor_uses_configured_paligemma_tokenizer(monkeypatch):
    tokenizer_path = "/home/phl/workspace/models/paligemma-tokenizer"
    tokenizer_steps = []

    class FakeTokenizerProcessorStep(ProcessorStep):
        def __init__(self, *, tokenizer_name, max_length, padding_side, padding):
            self.tokenizer_name = tokenizer_name
            self.max_length = max_length
            self.padding_side = padding_side
            self.padding = padding
            tokenizer_steps.append(self)

        def __call__(self, transition):
            return transition

        def transform_features(self, features):
            return features

    monkeypatch.setattr(
        "lerobot.policies.pi05.processor_pi05.TokenizerProcessorStep",
        FakeTokenizerProcessorStep,
    )

    config = PI05Config(paligemma_tokenizer_name=tokenizer_path)
    config.input_features, config.output_features = _pi05_features()

    make_pi05_pre_post_processors(config=config)

    assert tokenizer_steps[0].tokenizer_name == tokenizer_path


def test_tactile_token_encoder_outputs_configured_tokens():
    encoder = TactileTokenEncoder(
        encoder_type="cnn",
        input_shape=(12, 32),
        feature_dim=64,
        n_tokens=2,
        dropout=0.0,
    )

    tokens = encoder(torch.randn(3, 12, 32))

    assert tokens.shape == (3, 2, 64)
