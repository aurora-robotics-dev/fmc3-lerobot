#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_TACTILE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    paligemma_tokenizer_name: str = field(
        default_factory=lambda: os.environ.get(
            "LEROBOT_PALIGEMMA_TOKENIZER",
            "google/paligemma-3b-pt-224",
        )
    )
    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
            "TACTILE": NormalizationMode.MEAN_STD,
        }
    )

    # Tactile sensor configuration.
    use_tactile: bool = False
    tactile_encoder_type: str = "cnn"  # choices: ["cnn", "attention"]
    tactile_input_shape: tuple[int, int] = (12, 32)
    tactile_dropout: float = 0.3
    tactile_feature_dim: int = 256
    # None means auto-detect from input_features. Supported examples:
    # ["observation.tactile.right"], ["observation.tactile.left"], or both.
    tactile_features: list[str] | None = None
    n_tactile_tokens: int = 1

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.tactile_encoder_type not in ["cnn", "attention"]:
            raise ValueError(
                f"Invalid tactile_encoder_type: {self.tactile_encoder_type}. "
                "Expected one of ['cnn', 'attention']."
            )

        if len(tuple(self.tactile_input_shape)) != 2:
            raise ValueError(f"tactile_input_shape must be 2D, got {self.tactile_input_shape}")

        if self.n_tactile_tokens < 1:
            raise ValueError(f"n_tactile_tokens must be >= 1, got {self.n_tactile_tokens}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

        if not self.use_tactile:
            self.tactile_features = None
            return

        self.tactile_input_shape = tuple(self.tactile_input_shape)
        detected_tactile_features = self._detect_tactile_features()

        if self.tactile_features is None:
            self.tactile_features = detected_tactile_features
        else:
            missing = [key for key in self.tactile_features if key not in self.input_features]
            if missing:
                raise ValueError(f"Configured tactile_features are missing from input_features: {missing}")
            self.tactile_features = self._sort_tactile_features(self.tactile_features)

        for key in self.tactile_features:
            feature = self.input_features[key]
            if feature.type is not FeatureType.TACTILE:
                self.input_features[key] = PolicyFeature(type=FeatureType.TACTILE, shape=feature.shape)
            if tuple(feature.shape) != tuple(self.tactile_input_shape):
                raise ValueError(
                    f"Tactile feature {key!r} shape must match tactile_input_shape "
                    f"{self.tactile_input_shape}, got {feature.shape}"
                )

        if not self.tactile_features:
            raise ValueError(
                "use_tactile=True but no 2D tactile feature was found. "
                f"Expected {OBS_TACTILE}, {OBS_TACTILE}.left, or {OBS_TACTILE}.right "
                f"with shape {self.tactile_input_shape}."
            )

    def _detect_tactile_features(self) -> list[str]:
        tactile_keys = []
        for key, feature in self.input_features.items():
            if not self._is_tactile_key(key):
                continue
            if len(feature.shape) != 2:
                continue
            tactile_keys.append(key)
        return self._sort_tactile_features(tactile_keys)

    def _is_tactile_key(self, key: str) -> bool:
        return key == OBS_TACTILE or key.startswith(f"{OBS_TACTILE}.")

    def _sort_tactile_features(self, keys: list[str]) -> list[str]:
        priority = {
            OBS_TACTILE: 0,
            f"{OBS_TACTILE}.left": 1,
            f"{OBS_TACTILE}.right": 2,
        }
        return sorted(keys, key=lambda key: (priority.get(key, 100), key))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
