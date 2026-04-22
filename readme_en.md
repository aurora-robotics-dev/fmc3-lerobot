# fmc3-lerobot / Fourier GR2 LeRobot

This repository is a local Fourier GR2-oriented fork/extension of Hugging Face LeRobot. It keeps the standard LeRobot library layout while adding practical tooling for GR2 dataset conversion, replay, PI0 fine-tuning, real-robot deployment, RGB/RGB-D/wrist-camera inference, and Unix Socket inference services.

Before training or deployment, verify your conda environment, robot DDS domain, camera devices, and checkpoint paths on the target machine.

### Features

- Core LeRobot library for policies, datasets, robots, environments, training, and evaluation.
- Fourier GR2 data conversion from Dora-Record style episodes to LeRobot v3 format.
- GR2 dataset replay on real robots, with dry-run and safety controls.
- PI0 fine-tuning workflows for GR2 datasets.
- PI0 deployment scripts for RGB, RGB-D, and wrist-camera setups.
- Unix Domain Socket inference services for integration with RoboOS or external controllers.
- Environment notes and package snapshots for `lerobot` and `lerobot-pi0`.

### Repository Layout

```text
.
├── src/lerobot/                  # Core LeRobot package
├── tests/                        # pytest tests
├── docs/                         # documentation sources
├── examples/                     # runnable examples
├── Robot/fouier/                 # Fourier/GR2 data conversion tools
├── scripts/                      # GR2 training, deployment, inference, replay scripts
├── scripts/train/                # training scripts
├── scripts/inference/            # inference services and clients
├── gr2_env/                      # GR2 environment and configuration notes
├── fourier_aurora_sdk/           # Fourier Aurora SDK related files
├── outputs/                      # local outputs
└── Log/                          # local logs
```

### Installation

Use Python 3.10.

General LeRobot development, testing, and GR2 replay:

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -e ".[dev,test]"
pip install fourier-aurora-client
```

PI0 training, deployment, and inference services:

```bash
conda create -n lerobot-pi0 python=3.10
conda activate lerobot-pi0
pip install -e ".[pi,dev,test]"
pip install fourier-aurora-client
```

For a smaller PI0 runtime environment:

```bash
pip install -e ".[pi]"
pip install fourier-aurora-client
```

Validation:

```bash
python -c "import lerobot; print('lerobot import ok')"
python -c "import fourier_aurora_client; print('aurora import ok')"
lerobot-train --help
```

Extra PI0 validation:

```bash
python -c "from lerobot.policies.pi0.modeling_pi0 import PI0Policy; print('pi0 import ok')"
```

### GR2 Runtime Conventions

Keep dataset conversion, training, replay, and deployment aligned with the same conventions:

- `robot_type=fourier_gr2`
- `robot_name=gr2`
- `domain_id=123`
- `fsm_state=11`
- typical control rate: `30 FPS`
- typical camera size: `640x480`
- current standard state dimension: `state_dim=45`
- current standard action dimension: `action_dim=35`

Current 35D action layout:

| Index | Group | Dim |
| --- | --- | --- |
| `0:7` | left_manipulator | 7 |
| `7:14` | right_manipulator | 7 |
| `14:20` | left_hand | 6 |
| `20:26` | right_hand | 6 |
| `26:28` | head | 2 |
| `28:29` | waist yaw | 1 |
| `29:35` | base | 6 |

Common observation keys:

```text
observation.state
observation.images.camera_top
observation.images.camera_top_depth
```

Specialized wrist-camera scripts may add left/right wrist camera inputs.

### Dataset Conversion

Main converter:

```bash
Robot/fouier/convert_dora_to_lerobot.py
```

Example:

```bash
python Robot/fouier/convert_dora_to_lerobot.py \
  --input /path/to/dora_episode_root \
  --output /path/to/lerobot_output \
  --task "teleoperation task" \
  --fps 30 \
  --video-codec libx264 \
  --robot-type fourier_gr2
```

Expected LeRobot v3-style output:

```text
dataset_name/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── camera_top/
        └── chunk-000/
            └── file-000.mp4
```

### Dataset Replay

Use the `lerobot` environment for replay unless you have a specific reason to use another environment.

Dry-run validation:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/replay_gr2_dataset.py \
  --dataset-path /path/to/dataset \
  --episode 0 \
  --dry-run \
  --verbose
```

Real robot replay:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/replay_gr2_dataset.py \
  --dataset-path /path/to/dataset \
  --episode 0 \
  --domain-id 123 \
  --fps 30 \
  --transition-time 3.0
```

For the first robot run, keep clamping enabled and use a small `--max-joint-delta`.

### PI0 Training

Use the `lerobot-pi0` environment. Useful reference scripts:

```text
scripts/train/gr2_train.sh
scripts/tools/train_pi0_gr2_black_capped_bottle_yellow_to_green.sh
scripts/tools/train_pi0_gr2_grab_bottle_from_box_to_desk_rgb.sh
```

Generic training example:

```bash
conda run --no-capture-output -n lerobot-pi0 \
  lerobot-train \
  --dataset.repo_id=local/my_gr2_dataset \
  --dataset.root=/path/to/my_gr2_dataset \
  --dataset.streaming=false \
  --dataset.video_backend=torchcodec \
  --policy.path=/path/to/pi0 \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.max_state_dim=45 \
  --policy.max_action_dim=35 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/path/to/output \
  --job_name=my_gr2_run \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=50 \
  --batch_size=4 \
  --wandb.enable=true \
  --wandb.disable_artifact=false \
  --wandb.project=my_gr2_project
```

If GPU memory is limited, reduce `batch_size` first and keep `gradient_checkpointing=true`.

### Deployment And Inference

Common entry points:

```text
scripts/deploy_gr2_pi0.py
scripts/deploy_gr2_pi0_rgbd.py
scripts/deploy_gr2_pi0_rgb_right_wrist.py
scripts/gr2_pi0_inference_service.py
scripts/inference/gr2_dual_pi0_rgb_wrist_inference_server.py
```

Single-model deployment:

```bash
conda run --no-capture-output -n lerobot-pi0 \
  python scripts/deploy_gr2_pi0.py \
  --checkpoint-path /path/to/pretrained_model \
  --task "pick bottle" \
  --robot-type fourier_gr2 \
  --robot-name gr2 \
  --domain-id 123 \
  --fsm-state 11 \
  --fps 30 \
  --device auto
```

Unix Socket inference service:

```bash
CONDA_ENV=lerobot-pi0 \
CHECKPOINT_PATH=/path/to/pretrained_model \
TASK="pick bottle and place into box" \
bash scripts/inference/start_gr2_pi0_inference_service.sh
```

Dual-model RGB wrist inference service:

```bash
bash scripts/inference/start_gr2_dual_pi0_rgb_wrist_inference_server.sh
```

Client examples:

```bash
CLIENT="python scripts/inference/gr2_dual_pi0_rgb_wrist_client.py"
$CLIENT start-take-out --fps 15 --max-steps 0
$CLIENT start-put-in --fps 15 --max-steps 0
$CLIENT status
$CLIENT stop
```

### Testing And Quality

```bash
pre-commit install
pre-commit run --all-files
pytest tests -vv --maxfail=10
make test-end-to-end DEVICE=cpu
```

For faster iteration:

```bash
pytest tests/datasets/test_dataset_tools.py -vv
```

### Safety Notes

- Do not commit secrets, dataset credentials, checkpoints, or large generated artifacts.
- Always run `--dry-run` before controlling a real robot.
- Start with low speed, small action deltas, and an accessible emergency stop.
- Keep dataset conversion, training, and deployment dimensions aligned.
- Verify `domain_id`, `robot_name`, and `fsm_state` against the actual robot setup.

### More Documentation

- `gr2_env/ENVIRONMENT_USAGE_GUIDE.md`
- `gr2_env/FOURIER_GR2_CONFIG_SUMMARY.md`
- `gr2_env/FOURIER_GR2_TRAINING_CONFIG.md`
- `gr2_env/FOURIER_GR2_DEPLOYMENT_CONFIG.md`
- `gr2_env/FOURIER_GR2_DATASET_CONFIG.md`
- `scripts/README_replay_gr2.md`
- `scripts/inference/README.md`
- `scripts/train/README.md`
