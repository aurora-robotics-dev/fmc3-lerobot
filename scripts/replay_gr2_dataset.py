#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GR-2 机器人数据集回放脚本

从 LeRobot v3.0 数据集中读取 episode，并在 GR-2 机器人上回放动作序列。

功能特性：
- 支持本地和 HuggingFace Hub 数据集
- 精确帧率控制（默认 30fps）
- 动作平滑与安全限位
- 可视化回放进度
- 支持 dry-run 模式（无需连接真实机器人）

使用示例：
    # 回放本地数据集
    python scripts/replay_gr2_dataset.py \\
        --dataset-path ./outputs/dataset/gr2_pick_bottle \\
        --episode 0 \\
        --domain-id 123

    # 回放 HuggingFace 数据集
    python scripts/replay_gr2_dataset.py \\
        --repo-id username/gr2-pick-bottle \\
        --episode 0 \\
        --domain-id 123

    # Dry-run 模式（不连接机器人）
    python scripts/replay_gr2_dataset.py \\
        --dataset-path ./outputs/dataset/gr2_pick_bottle \\
        --episode 0 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# LeRobot 数据集加载
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.gr2_hand_conversion import hand_sdk_to_urdf, hand_urdf_to_sdk

# GR-2 机器人控制（需要 fourier_aurora_client）
try:
    from fourier_aurora_client import AuroraClient
    AURORA_AVAILABLE = True
except ImportError:
    AURORA_AVAILABLE = False
    logging.warning("fourier_aurora_client not found. Only dry-run mode available.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# GR-2 动作维度定义（URDF 空间，35D）
# ============================================================================
# 0:7   - left_manipulator (左臂 7 关节)
# 7:14  - right_manipulator (右臂 7 关节)
# 14:20 - left_hand (左手 6 关节)
# 20:26 - right_hand (右手 6 关节)
# 26:28 - head (头部 2 关节: yaw, pitch)
# 28:29 - waist (腰部 1 关节: yaw)
# 29:35 - base (底盘 6D: x, y, z, roll, pitch, yaw)
# ============================================================================

# GR-2 关节组 → 动作数组切片映射（与 deploy_gr2_pi0.py 一致）
ACTION_GROUP_SLICES = {
    "left_manipulator": slice(0, 7),
    "right_manipulator": slice(7, 14),
    "left_hand": slice(14, 20),
    "right_hand": slice(20, 26),
    "head": slice(26, 28),
    "waist": slice(28, 29),
}

GROUP_DIMS = {name: s.stop - s.start for name, s in ACTION_GROUP_SLICES.items()}

# 真实关节限位（URDF 空间，来自 deploy_gr2_pi0.py）
JOINT_LIMITS = {
    "left_manipulator": [
        (-2.9671, 2.9671), (-0.5236, 2.7925), (-1.8326, 1.8326),
        (-1.5272, 0.47997), (-1.8326, 1.8326), (-0.61087, 0.61087), (-0.95993, 0.95993),
    ],
    "right_manipulator": [
        (-2.9671, 2.9671), (-2.7925, 0.5236), (-1.8326, 1.8326),
        (-1.5272, 0.47997), (-1.8326, 1.8326), (-0.61087, 0.61087), (-0.95993, 0.95993),
    ],
    "head": [(-1.3963, 1.3963), (-0.5236, 0.5236)],
    "waist": [(-2.618, 2.618)],
}

# FSM 状态常量
FSM_PD_STAND = 2          # PD 站立（可控制上半身）
FSM_UPPER_BODY_CMD = 11   # 上半身用户指令模式


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Replay LeRobot dataset on Fourier GR-2 robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        help="本地数据集路径（与 --repo-id 互斥）"
    )
    dataset_group.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace 数据集 ID，格式: username/dataset-name"
    )

    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="要回放的 episode 索引"
    )

    # 机器人参数
    parser.add_argument(
        "--domain-id",
        type=int,
        default=123,
        help="Aurora SDK DDS 域 ID"
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="gr2",
        help="机器人名称"
    )

    # 回放参数
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="回放帧率（Hz）"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="起始帧索引"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最大回放帧数（0 表示回放整个 episode）"
    )

    # 安全参数
    parser.add_argument(
        "--disable-clamp",
        action="store_true",
        help="禁用关节限位夹紧（危险！）"
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.1,
        help="单步最大关节变化量（rad），用于平滑"
    )
    parser.add_argument(
        "--transition-time",
        type=float,
        default=3.0,
        help="首帧平滑过渡时间（秒）"
    )

    # 调试参数
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run 模式：不连接机器人，仅打印动作"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出"
    )

    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> LeRobotDataset:
    """加载 LeRobot 数据集"""
    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        LOGGER.info(f"Loading local dataset: {dataset_path}")
        dataset = LeRobotDataset(str(dataset_path))
    else:
        LOGGER.info(f"Loading HuggingFace dataset: {args.repo_id}")
        dataset = LeRobotDataset(args.repo_id)

    LOGGER.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")
    LOGGER.info(f"Dataset FPS: {dataset.fps}")
    LOGGER.info(f"Action features: {dataset.features[ACTION]}")

    return dataset


def filter_episode(dataset: LeRobotDataset, episode_idx: int) -> tuple[list[np.ndarray], int]:
    """从数据集中提取指定 episode 的所有动作帧

    Args:
        dataset: LeRobot 数据集对象
        episode_idx: Episode 索引

    Returns:
        (actions, num_frames): 动作列表和帧数
    """
    LOGGER.info(f"Filtering episode {episode_idx}...")

    # 过滤出指定 episode 的所有帧
    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == episode_idx
    )

    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode_idx} not found in dataset")

    # 提取动作列
    actions_column = episode_frames.select_columns(ACTION)
    action_names = dataset.features[ACTION]["names"]

    LOGGER.info(f"Episode {episode_idx}: {len(episode_frames)} frames")
    LOGGER.info(f"Action dimension: {len(action_names)}")
    LOGGER.info(f"Action names: {action_names}")

    # 将动作数组转换为字典列表
    actions = []
    for idx in range(len(episode_frames)):
        action_array = actions_column[idx][ACTION]
        action_dict = {
            name: float(action_array[i])
            for i, name in enumerate(action_names)
        }
        actions.append(action_dict)

    return actions, len(episode_frames)


def clamp_joint_action(action: np.ndarray) -> np.ndarray:
    """夹紧非手部关节动作到安全范围（使用真实 GR-2 关节限位）

    手部关节不做夹紧，因为它们在 URDF 空间，后续会通过 hand_urdf_to_sdk 转换。
    """
    action = action.copy()
    for group_name, limits in JOINT_LIMITS.items():
        s = ACTION_GROUP_SLICES[group_name]
        for i, (lo, hi) in enumerate(limits):
            if s.start + i < len(action):
                action[s.start + i] = np.clip(action[s.start + i], lo, hi)
    return action


def stabilize_action(
    current_action: np.ndarray,
    prev_action: np.ndarray,
    max_delta: float
) -> np.ndarray:
    """限制步间动作变化量，使运动更平滑

    Args:
        current_action: 当前动作
        prev_action: 上一步动作
        max_delta: 最大变化量（rad）

    Returns:
        平滑后的动作
    """
    delta = current_action - prev_action
    delta = np.clip(delta, -max_delta, max_delta)
    return prev_action + delta


def action_dict_to_array(action_dict: dict[str, float], expected_dim: int = 35) -> np.ndarray:
    """将动作字典转换为 35D 数组（GR-2 URDF 空间）

    Args:
        action_dict: 动作字典，键为关节名称
        expected_dim: 期望的动作维度

    Returns:
        35D 动作数组
    """
    # 按照 GR-2 URDF 顺序排列
    action_array = np.zeros(expected_dim, dtype=np.float32)

    # 填充已知的关节值
    for i, (key, value) in enumerate(action_dict.items()):
        if i < expected_dim:
            action_array[i] = value

    return action_array


def send_action_to_robot(
    client: Any,
    action: np.ndarray,
    send_base: bool = False
) -> None:
    """将动作发送到 GR-2 机器人

    与 deploy_gr2_pi0.py 的 send_action_to_robot 逻辑一致：
    1. 手部关节从 URDF 空间转换到 SDK 空间
    2. 用 set_joint_positions(dict) 一次性下发所有关节组
    """
    action_send = action.copy()

    # 手部 URDF → SDK 坐标转换
    action_send[14:20] = hand_urdf_to_sdk(action_send[14:20], clip=True)
    action_send[20:26] = hand_urdf_to_sdk(action_send[20:26], clip=True)

    # 构建关节组字典，一次性下发
    position_dict = {
        group_name: action_send[group_slice].tolist()
        for group_name, group_slice in ACTION_GROUP_SLICES.items()
    }
    client.set_joint_positions(position_dict)

    # 底盘速度控制（可选）
    if send_base and len(action) >= 32:
        try:
            vel_x = float(action[29])
            vel_y = float(action[30])
            vel_yaw = float(action[31]) if len(action) > 31 else 0.0
            client.set_velocity(vel_x, vel_y, vel_yaw)
        except Exception:
            pass


def smooth_transition(
    client: Any,
    target_action: np.ndarray,
    duration: float,
    freq: float = 30.0
) -> None:
    """从当前位置平滑过渡到目标动作（与 deploy_gr2_pi0.py 逻辑一致）"""
    LOGGER.info(f"Smooth transition to first action ({duration:.1f}s)...")

    if duration <= 0:
        send_action_to_robot(client, target_action, send_base=False)
        return

    # 获取当前关节位置（URDF 空间）
    current_action = np.zeros(29, dtype=np.float32)
    try:
        for group_name, group_slice in ACTION_GROUP_SLICES.items():
            pos = client.get_group_state(group_name, "position")
            if pos is not None:
                arr = np.asarray(pos, dtype=np.float32)
                # 手部 SDK → URDF 转换
                if "hand" in group_name and arr.shape[0] == 6:
                    arr = hand_sdk_to_urdf(arr)
                current_action[group_slice] = arr
    except Exception as e:
        LOGGER.warning(f"Failed to get current state: {e}. Using zero as start.")

    # 线性插值（只插值 29 个关节，不含底盘）
    target_joints = target_action[:29]
    num_steps = max(1, int(duration * freq))
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated = current_action * (1.0 - alpha) + target_joints * alpha
        send_action_to_robot(client, interpolated, send_base=False)
        time.sleep(1.0 / freq)

    LOGGER.info("Transition complete.")


def replay_episode(
    actions: list[dict[str, float]],
    args: argparse.Namespace,
    client: Any | None = None
) -> None:
    """回放 episode 动作序列

    Args:
        actions: 动作字典列表
        args: 命令行参数
        client: Aurora 客户端（dry-run 模式下为 None）
    """
    num_frames = len(actions)
    start_frame = args.start_frame
    max_frames = args.max_frames if args.max_frames > 0 else num_frames

    LOGGER.info(f"Starting replay: frames {start_frame} to {min(start_frame + max_frames, num_frames)}")
    LOGGER.info(f"Target FPS: {args.fps:.1f}")

    prev_action = None
    first_action = True

    for idx in range(start_frame, min(start_frame + max_frames, num_frames)):
        loop_start = time.perf_counter()

        # 转换动作格式
        action_dict = actions[idx]
        action_array = action_dict_to_array(action_dict, expected_dim=35)

        # 安全处理
        if not args.disable_clamp:
            action_array = clamp_joint_action(action_array)

        if prev_action is not None:
            action_array = stabilize_action(action_array, prev_action, args.max_joint_delta)

        # 发送动作
        if not args.dry_run:
            if client is None:
                raise RuntimeError("Robot client is not initialized")

            if first_action and args.transition_time > 0:
                smooth_transition(client, action_array, args.transition_time)
                first_action = False
            else:
                send_action_to_robot(client, action_array, send_base=False)
                first_action = False

        # 日志输出
        if args.verbose or idx % 30 == 0:
            arm_norm = np.linalg.norm(action_array[0:14])
            hand_norm = np.linalg.norm(action_array[14:26])
            LOGGER.info(
                f"Frame {idx}/{num_frames} | "
                f"Arm: {arm_norm:.3f} | Hand: {hand_norm:.3f}"
            )

        prev_action = action_array.copy()

        # 帧率控制
        elapsed = time.perf_counter() - loop_start
        sleep_time = max(0.0, 1.0 / args.fps - elapsed)
        time.sleep(sleep_time)

    LOGGER.info("Replay complete.")


def main() -> None:
    """主函数"""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查依赖
    if not args.dry_run and not AURORA_AVAILABLE:
        LOGGER.error("fourier_aurora_client not found. Use --dry-run or install the SDK.")
        sys.exit(1)

    # 加载数据集
    try:
        dataset = load_dataset(args)
        actions, num_frames = filter_episode(dataset, args.episode)
    except Exception as e:
        LOGGER.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 连接机器人
    client = None
    if not args.dry_run:
        try:
            LOGGER.info(f"Connecting to GR-2 (domain_id={args.domain_id})...")
            client = AuroraClient.get_instance(
                domain_id=args.domain_id,
                robot_name=args.robot_name
            )
            time.sleep(1.0)

            # 切换到上半身控制模式
            LOGGER.info("Switching to upper body control mode...")
            client.set_fsm_state(FSM_UPPER_BODY_CMD)
            time.sleep(1.0)

            LOGGER.info("Robot connected.")
        except Exception as e:
            LOGGER.error(f"Failed to connect to robot: {e}")
            sys.exit(1)
    else:
        LOGGER.info("Dry-run mode: robot connection skipped.")

    # 回放 episode
    try:
        replay_episode(actions, args, client)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Replay failed: {e}", exc_info=True)
    finally:
        # 清理
        if client is not None:
            try:
                LOGGER.info("Disconnecting robot...")
                client.close()
            except Exception:
                pass

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
