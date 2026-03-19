# -*- coding: utf-8 -*-
"""
部署 ACT 策略到傅里叶 GR-2 机器人
相机: Orbbec Gemini 335Lg (RGB via OpenCV, 深度零填充)

用法:
    python scripts/deploy_gr2_act.py \
        --checkpoint outputs/train/fourier_gr2_act/checkpoints/last/pretrained_model \
        --domain-id 123 --robot-name gr2

    # 调试模式 (不连接机器人)
    python scripts/deploy_gr2_act.py \
        --checkpoint outputs/train/fourier_gr2_act/checkpoints/last/pretrained_model \
        --dry-run
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deploy_gr2_act")

# =====================================================================
# Action 35维 → GR2Robot 关节组映射
# 顺序与 GR2_JOINT_ORDER 一致 (convert_dora_to_lerobot.py 已重排到 SDK 顺序)
# left_arm(7) + right_arm(7) + left_hand(6) + right_hand(6) + head(2) + waist(1) + base_vel(6)
# =====================================================================
ACTION_SLICES = {
    "left_manipulator":  slice(0, 7),
    "right_manipulator": slice(7, 14),
    "left_hand":         slice(14, 20),
    "right_hand":        slice(20, 26),
    "head":              slice(26, 28),
    "waist":             slice(28, 29),   # GR-2 只有 waist_yaw
}
BASE_VEL_SLICE = slice(29, 35)


class GR2PolicyDeployer:
    """将 LeRobot ACT 策略部署到傅里叶 GR-2 机器人"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        domain_id: int = 123,
        robot_name: str = "gr2",
        camera_ids: list[int] | None = None,
        control_freq: int = 30,
        dry_run: bool = False,
        enable_base: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.control_freq = control_freq
        self.dry_run = dry_run
        self.enable_base = enable_base
        self.running = False
        self.camera_ids = camera_ids or [6]

        logger.info(f"设备: {self.device}")
        logger.info(f"Checkpoint: {checkpoint_path}")

        self._load_policy(checkpoint_path)
        self._load_processors(checkpoint_path)
        self._init_cameras()

        if not self.dry_run:
            self._init_robot(domain_id, robot_name)
        else:
            self.robot = None
            logger.info("[DRY RUN] 跳过机器人连接")

    def _load_policy(self, checkpoint_path: str):
        from lerobot.policies.act.modeling_act import ACTPolicy
        logger.info("正在加载 ACT 策略...")
        self.policy = ACTPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)
        self.policy.eval()
        logger.info(f"策略加载完成 (chunk_size={self.policy.config.chunk_size}, "
                     f"n_action_steps={self.policy.config.n_action_steps})")

    def _load_processors(self, checkpoint_path: str):
        from lerobot.policies.factory import make_pre_post_processors
        logger.info("正在加载预处理器/后处理器...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint_path,
        )
        logger.info("预处理器/后处理器加载完成")

    def _init_cameras(self):
        self.cap_rgb = None
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ret, frame = cap.read()
                if ret and frame.shape == (480, 640, 3):
                    self.cap_rgb = cap
                    logger.info(f"Orbbec RGB 相机已打开 (/dev/video{cam_id})")
                    break
                cap.release()
        if self.cap_rgb is None:
            logger.warning("未找到可用的 RGB 相机")

    def _init_robot(self, domain_id: int, robot_name: str):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Robot"))
        from gr2_robot import GR2Robot
        logger.info(f"正在连接 GR-2 (domain_id={domain_id}, name={robot_name})...")
        self.robot = GR2Robot(domain_id=domain_id, robot_name=robot_name)
        logger.info("机器人连接成功")

    def get_camera_images(self) -> tuple[np.ndarray, np.ndarray]:
        if self.cap_rgb is not None:
            ret, frame = self.cap_rgb.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return rgb, np.zeros((480, 640, 3), dtype=np.uint8)
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        return black, black

    def get_robot_state(self) -> np.ndarray:
        if self.dry_run:
            return np.zeros(45, dtype=np.float32)

        state = np.zeros(45, dtype=np.float32)
        # 顺序与 GR2_JOINT_ORDER 一致 (convert_dora_to_lerobot.py 的重排顺序)
        # left_arm(7) + right_arm(7) + left_hand(6) + right_hand(6) + head(2) + waist(1) = 29
        # + base_state(16) = 45
        group_map = {
            "left_manipulator":  slice(0, 7),
            "right_manipulator": slice(7, 14),
            "left_hand":         slice(14, 20),
            "right_hand":        slice(20, 26),
            "head":              slice(26, 28),
            "waist":             slice(28, 29),
        }
        for group_name, s in group_map.items():
            try:
                positions = self.robot.get_joint_positions(group_name)
                if positions:
                    state[s] = positions[:s.stop - s.start]
            except Exception as e:
                logger.warning(f"读取 {group_name} 失败: {e}")

        try:
            imu = self.robot.get_base_imu()
            # base_state(16D): pos(3) + quat(4) + rpy(3) + acc(3) + omega(3)
            # 从 index 29 开始
            quat = imu.get("orientation", [0, 0, 0, 1])
            state[32:36] = quat[:4]   # base_quat: 29+3=32
            acc = imu.get("acceleration", [0, 0, 0])
            state[39:42] = acc[:3]    # imu_acc: 29+10=39
            omega = imu.get("angular_vel", [0, 0, 0])
            state[42:45] = omega[:3]  # imu_omega: 29+13=42
        except Exception as e:
            logger.warning(f"读取 IMU 失败: {e}")

        return state

    def build_observation(self) -> dict[str, torch.Tensor]:
        state = self.get_robot_state()
        rgb, depth = self.get_camera_images()
        return {
            "observation.state": torch.from_numpy(state).float(),
            "observation.images.camera_top": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
            "observation.images.camera_top_depth": torch.from_numpy(depth).permute(2, 0, 1).float() / 255.0,
        }

    def action_to_robot_command(self, action: np.ndarray):
        # 数据集已通过 reorder_to_target() 排成 SDK 顺序，直接发送
        joint_targets = {}
        for group_name, s in ACTION_SLICES.items():
            joint_targets[group_name] = action[s].tolist()

        base_vel = action[BASE_VEL_SLICE]

        if self.dry_run:
            logger.info(f"[DRY RUN] 关节: { {k: [round(v, 3) for v in vals] for k, vals in joint_targets.items()} }")
            logger.info(f"[DRY RUN] 基座: vel_x={base_vel[0]:.3f}, vel_y={base_vel[1]:.3f}, vel_yaw={base_vel[2]:.3f}")
            return

        self.robot.client.set_joint_positions(joint_targets)

        if self.enable_base:
            self.robot.set_velocity(
                vx=float(base_vel[0]),
                vy=float(base_vel[1]),
                vyaw=float(base_vel[2]),
            )

    def prepare_robot(self):
        if self.dry_run:
            logger.info("[DRY RUN] 跳过机器人准备")
            return

        input("请确保机器人周围安全，按回车切换到上半身控制模式...")
        self.robot.upper_body_mode()
        time.sleep(1.0)
        logger.info("已切换到上半身控制模式")

        # 复位到训练数据的初始姿态
        input("按回车复位到初始姿态 (3秒平滑移动)...")
        init_pose = {
            "left_manipulator":  [0.012, 0.167, 0.051, -0.711, 0.054, -0.139, -0.006],
            "right_manipulator": [-0.092, -0.006, -0.241, 0.070, 0.089, -0.409, -0.200],
            "head":              [-0.007, 0.013],
            "waist":             [-0.008, 0.0, 0.0],
            "left_hand":         [-0.012, -0.012, -0.005, -0.009, 0.0, -1.578],
            "right_hand":        [-0.017, -0.017, -0.008, -0.008, 0.006, -1.569],
        }
        self.robot.move_joints(init_pose, duration=3.0, frequency=100)
        logger.info("已复位到初始姿态")

        if self.enable_base:
            input("按回车切换到行走模式...")
            self.robot.walk_mode()
            time.sleep(1.0)

        input("按回车开始策略推理...")

    def run(self, max_steps: int = 3000):
        self.running = True
        dt = 1.0 / self.control_freq

        def signal_handler(sig, frame):
            logger.info("\n收到中断信号，正在安全停止...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        self.prepare_robot()
        self.policy.reset()

        logger.info(f"开始推理 (频率: {self.control_freq}Hz, 最大步数: {max_steps})")
        logger.info("按 Ctrl+C 安全停止")

        step = 0
        try:
            for step in range(max_steps):
                if not self.running:
                    break

                t_start = time.time()

                obs = self.build_observation()
                obs_processed = self.preprocessor(obs)

                with torch.no_grad():
                    action_tensor = self.policy.select_action(obs_processed)

                action_tensor = self.postprocessor(action_tensor)
                action = action_tensor.squeeze(0).numpy()

                self.action_to_robot_command(action)

                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step % 30 == 0:
                    actual_freq = 1.0 / max(elapsed, 1e-6)
                    logger.info(f"步骤 {step}/{max_steps}, "
                                f"耗时 {elapsed*1000:.1f}ms, "
                                f"频率 {actual_freq:.1f}Hz")

        except Exception as e:
            logger.error(f"推理异常: {e}", exc_info=True)
        finally:
            logger.info(f"推理结束, 共 {step} 步")
            self.stop()

    def stop(self):
        self.running = False
        if self.robot is not None:
            try:
                self.robot.set_velocity(0, 0, 0)
                logger.info("机器人已停止")
            except Exception as e:
                logger.warning(f"停止出错: {e}")

    def cleanup(self):
        if self.cap_rgb is not None:
            self.cap_rgb.release()
        if self.robot is not None:
            self.robot.close()
        logger.info("资源已释放")


def main():
    parser = argparse.ArgumentParser(description="部署 ACT 策略到傅里叶 GR-2")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/train/fourier_gr2_act/checkpoints/last/pretrained_model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--robot-name", type=str, default="gr2")
    parser.add_argument("--camera-ids", type=int, nargs="+", default=[6])
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-base", action="store_true")
    args = parser.parse_args()

    deployer = GR2PolicyDeployer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        domain_id=args.domain_id,
        robot_name=args.robot_name,
        camera_ids=args.camera_ids,
        control_freq=args.control_freq,
        dry_run=args.dry_run,
        enable_base=args.enable_base,
    )

    try:
        deployer.run(max_steps=args.max_steps)
    finally:
        deployer.cleanup()


if __name__ == "__main__":
    main()
