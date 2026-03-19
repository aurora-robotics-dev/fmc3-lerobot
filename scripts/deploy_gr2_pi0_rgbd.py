#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在 Fourier GR2 机器人上部署 PI0 策略 —— 多视觉输入版本（RGB + 深度图）。

本脚本基于单视觉输入版本 `scripts/deploy_gr2_pi0.py`（下文称 base），
保留了其安全/控制循环逻辑，同时解决了一个关键问题：
  当 PI0 模型的配置中声明了多个视觉输入键（例如同时需要
  `observation.images.camera_top` 和 `observation.images.camera_top_depth`）时，
  base 脚本只喂了 RGB 一路，本脚本会在每一步把 RGB 与深度图同时喂给策略。

整体流程（每个控制步）：
  1. 从 Orbbec RGB-D 相机读取 RGB + 深度帧
  2. 从 Aurora SDK 获取 GR2 机器人状态（关节角度等 45D 向量）
  3. 将 RGB、深度图、状态按策略要求的多视觉键打包成 observation
  4. 经过 preprocessor → PI0 推理 → postprocessor 得到动作向量
  5. 对动作进行夹紧(clamp)、平滑(stabilize)处理
  6. 下发到机器人执行
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 导入单视觉版本脚本作为 base，复用其中的工具函数：
#   - parse_args()            : 命令行参数解析
#   - OrbbecRGBDCamera        : Orbbec 深度相机封装
#   - resolve_pretrained_model_dir / load_train_config : 模型路径与配置解析
#   - select_device           : 选择 GPU/CPU 设备
#   - infer_policy_io         : 从策略配置中推断 state/action/visual 的 key 与维度
#   - fit_vector              : 将向量裁剪或零填充到目标维度
#   - clamp_non_hand_joint_action : 夹紧非手部关节动作到安全范围
#   - stabilize_action        : 动作平滑（限制步间变化量）
#   - smooth_transition       : 首帧平滑过渡
#   - send_action_to_robot    : 下发 URDF 空间动作到机器人
#   - get_robot_state_urdf    : 获取 URDF 空间下的机器人状态
#   - setup_gui / render_gui_frame / save_snapshot : GUI 可视化与快照保存
#   - setup_robot_if_needed   : 连接机器人客户端
import deploy_gr2_pi0 as base
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference

LOGGER = logging.getLogger(__name__)
# 标记是否收到系统终止信号（SIGINT / SIGTERM），用于优雅退出
_RECEIVED_STOP_SIGNAL = False


def _ordered_visual_keys(visual_keys: list[str], preferred_key: str) -> list[str]:
    """对视觉键列表排序，确保 preferred_key（通常是主 RGB 相机）排在最前面。

    这样做的目的是保证在构建 observation 时，主相机图像永远是第一个处理的，
    在部分只取首个视觉键的代码路径下也能正确运行。

    Args:
        visual_keys: 策略配置中声明的所有视觉输入键，
                     例如 ["observation.images.camera_top", "observation.images.camera_top_depth"]
        preferred_key: 用户通过 --camera-key 指定的首选键名

    Returns:
        重新排列后的视觉键列表，preferred_key 在最前（如果存在的话）
    """
    if preferred_key in visual_keys:
        return [preferred_key] + [k for k in visual_keys if k != preferred_key]
    return list(visual_keys)


def _build_visual_observation(
    visual_keys: list[str],
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
) -> dict[str, np.ndarray]:
    """根据策略要求的视觉键，构建视觉观测字典。

    分配规则很简单：
      - 键名中包含 "depth" → 使用深度伪彩色图 (depth_rgb)
      - 其他键               → 使用 RGB 图像

    这样一个 Orbbec RGB-D 相机就能同时满足 RGB 和深度两路输入需求。

    Args:
        visual_keys: 策略要求的所有视觉键名列表
        rgb:         Orbbec 相机读取的 RGB 图像, shape=(H, W, 3), dtype=uint8
        depth_rgb:   深度图转换为 3 通道伪彩色图像, shape=(H, W, 3), dtype=uint8

    Returns:
        dict: {视觉键名: 对应图像 ndarray}
    """
    obs: dict[str, np.ndarray] = {}
    for key in visual_keys:
        if "depth" in key.lower():
            obs[key] = depth_rgb
        else:
            obs[key] = rgb
    return obs


def warmup_policy_multivis(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    visual_keys: list[str],
    state_key: str,
    camera_height: int,
    camera_width: int,
    state_dim: int,
) -> None:
    """用全零假数据做一次完整的前向传播，触发模型的 JIT/CUDA 懒初始化。

    PI0 策略在首次推理时会有额外开销（CUDA kernel 编译、内存分配等），
    在正式控制循环开始前做一次 warmup 可以避免首帧延迟过高导致机器人抖动。

    与 base 脚本的区别：这里会为 **所有** 视觉键都构建假图像，
    而不仅仅是一个 RGB 键，确保多视觉输入路径也被预热到。

    Args:
        policy:        已加载的 PI0 策略模型
        preprocessor:  数据预处理管线（归一化、设备转移等）
        postprocessor: 数据后处理管线（反归一化等）
        device:        推理设备 (cuda / cpu)
        task:          自然语言任务描述，如 "pick bottle"
        robot_type:    机器人类型标识，如 "fourier_gr2"
        visual_keys:   策略需要的所有视觉输入键名列表
        state_key:     状态向量的键名，如 "observation.state"
        camera_height: 相机图像高度
        camera_width:  相机图像宽度
        state_dim:     状态向量维度（如 45）
    """
    # 构造全零假图像和假状态向量
    dummy_rgb = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    dummy_depth = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

    # 按多视觉键规则打包假 observation
    dummy_obs = _build_visual_observation(visual_keys, dummy_rgb, dummy_depth)
    dummy_obs[state_key] = np.zeros((state_dim,), dtype=np.float32)

    # 执行一次完整的推理管线: observation → batch → policy → action
    with torch.inference_mode():
        dummy_batch = prepare_observation_for_inference(
            observation=dummy_obs,
            device=device,
            task=task,
            robot_type=robot_type,
        )
        _ = postprocessor(policy.select_action(preprocessor(dummy_batch)))


def infer_single_action_multivis(
    policy: PI0Policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
    visual_keys: list[str],
    state_key: str,
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    state_model: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    """执行一次 PI0 多视觉输入推理，返回单步动作向量。

    推理管线：
      1. 将 RGB 和深度图按视觉键映射规则打包成 observation dict
      2. 加入状态向量
      3. prepare_observation_for_inference 将 numpy 数据转为 batch tensor
      4. preprocessor 做归一化 → policy.select_action 做前向推理 → postprocessor 做反归一化
      5. 从输出 tensor 中提取第一个动作（PI0 可能输出 action chunk），
         并用 fit_vector 裁剪/填充到目标 action_dim

    Args:
        policy / preprocessor / postprocessor / device: 同 warmup
        task / robot_type: 同 warmup
        visual_keys: 策略需要的所有视觉键名
        state_key:   状态向量键名
        rgb:         当前帧 RGB 图像 (H, W, 3) uint8
        depth_rgb:   当前帧深度伪彩色图像 (H, W, 3) uint8
        state_model: 当前机器人状态向量 (state_dim,) float32
        action_dim:  目标动作维度（如 35）

    Returns:
        np.ndarray: 单步动作向量 (action_dim,) float32
    """
    # 构建包含所有视觉输入和状态的完整 observation
    observation = _build_visual_observation(visual_keys, rgb, depth_rgb)
    observation[state_key] = state_model

    # 在 inference_mode 下执行推理（禁用梯度计算，节省显存和加速）
    with torch.inference_mode():
        batch = prepare_observation_for_inference(
            observation=observation,
            device=device,
            task=task,
            robot_type=robot_type,
        )
        # preprocessor: 归一化状态/图像 → policy: PI0 前向推理 → postprocessor: 反归一化动作
        action_tensor = postprocessor(policy.select_action(preprocessor(batch)))

    # 返回完整 action chunk (N, action_dim)，由主循环逐步消费
    all_actions = action_tensor.reshape(-1, action_tensor.shape[-1]).detach().cpu().numpy().astype(np.float32)
    return np.array([base.fit_vector(a, action_dim) for a in all_actions])


def run(args) -> None:
    """主部署流程：初始化模型/相机/机器人 → 进入控制循环 → 清理退出。

    整体分为三个阶段：
      (A) 初始化阶段：加载模型、创建 preprocessor/postprocessor、连接相机和机器人、warmup
      (B) 控制循环：循环执行「读取传感器 → 推理 → 下发动作 → 可视化」
      (C) 清理阶段：关闭相机、销毁 GUI 窗口、断开机器人连接
    """

    # =====================================================================
    # (A) 初始化阶段
    # =====================================================================

    # --- 解析模型路径和训练配置 ---
    pretrained_dir = base.resolve_pretrained_model_dir(args.checkpoint_path)
    train_cfg = base.load_train_config(pretrained_dir)
    device = base.select_device(args.device)

    LOGGER.info("Model: %s", pretrained_dir)
    LOGGER.info("Device: %s", device)
    if train_cfg:
        train_ds = train_cfg.get("dataset", {})
        LOGGER.info("Train dataset from config: repo_id=%s root=%s", train_ds.get("repo_id"), train_ds.get("root"))
    LOGGER.info("Camera target profile: %dx%d@%d", args.camera_width, args.camera_height, args.camera_fps)

    # --- 加载 PI0 策略模型 ---
    # strict=False: 允许 checkpoint 和模型定义之间有不匹配的 key（PI0 不同版本间常见）
    policy = PI0Policy.from_pretrained(str(pretrained_dir), strict=False).to(device)
    policy.eval()  # 切换到推理模式（关闭 dropout 等）

    # --- 创建数据预处理/后处理管线 ---
    # preprocessor: 对 observation 做归一化、图像裁剪缩放、设备转移等
    # postprocessor: 对 action 做反归一化
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # --- 从策略配置中推断 I/O 规格 ---
    # state_key:      状态键名，如 "observation.state"
    # cfg_state_dim:  模型配置中的状态维度
    # visual_keys:    所有视觉输入键名列表（本脚本的核心：支持多个）
    # action_key:     动作键名，如 "action"
    # cfg_action_dim: 模型配置中的动作维度
    state_key, cfg_state_dim, visual_keys, action_key, cfg_action_dim = base.infer_policy_io(policy)
    # 允许用户通过命令行覆盖维度（0 表示使用模型默认值）
    state_dim = args.state_dim if args.state_dim > 0 else cfg_state_dim
    action_dim = args.action_dim if args.action_dim > 0 else cfg_action_dim
    # 将用户指定的 camera_key 排到视觉键列表最前面
    visual_keys = _ordered_visual_keys(visual_keys, args.camera_key)

    LOGGER.info(
        "Policy IO: state_key=%s state_dim=%d action_key=%s action_dim=%d visual_keys=%s",
        state_key,
        state_dim,
        action_key,
        action_dim,
        visual_keys,
    )

    # --- 初始化 GUI 可视化 ---
    gui_enabled = base.setup_gui(args)

    # --- 配置快照保存目录 ---
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    last_vis_ts = time.time()
    frame_idx = 0
    periodic_save_every = max(0, int(args.save_every))
    # 如果 GUI 不可用且用户没有禁用 GUI，自动开启定期快照保存作为替代
    if not gui_enabled and not args.no_gui and periodic_save_every <= 0:
        periodic_save_every = 30
        LOGGER.info(
            "GUI unavailable. Fallback: save RGB/depth snapshots every %d steps to %s",
            periodic_save_every,
            save_dir,
        )

    # --- 初始化 Orbbec RGB-D 相机 ---
    camera = base.OrbbecRGBDCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        timeout_ms=args.camera_timeout_ms,
        warmup_frames=args.camera_warmup_frames,
        init_retries=args.camera_init_retries,
        retry_interval_s=args.camera_retry_interval_s,
    )
    camera.connect()

    # --- 连接 GR2 机器人（dry_run 模式下返回 None）---
    client = base.setup_robot_if_needed(args)

    # --- 模型预热：用假数据跑一次推理，消除首帧延迟 ---
    LOGGER.info("Warming up policy...")
    warmup_policy_multivis(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        task=args.task,
        robot_type=args.robot_type,
        visual_keys=visual_keys,
        state_key=state_key,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        state_dim=state_dim,
    )
    LOGGER.info("Warm-up complete.")

    # =====================================================================
    # (B) 主控制循环
    # =====================================================================

    step = 0
    first_action = True               # 标记是否是第一个动作（需要平滑过渡）
    policy.reset()                     # 重置策略内部状态（action chunk 缓存等）
    prev_action_urdf: np.ndarray | None = None  # 上一步动作，用于平滑
    no_frame_count = 0                 # 连续无帧计数（相机诊断用）
    action_chunk: np.ndarray | None = None  # 缓存的 action chunk (N, action_dim)
    chunk_idx = 0                      # 当前消费到 chunk 的第几步
    no_frame_last_log_ts = time.time() # 上次无帧日志时间（避免刷屏）
    slow_loop_last_log_ts = time.time()  # 上次慢循环告警时间

    try:
        while True:
            loop_start = time.perf_counter()

            # --- Step 1: 读取相机帧 ---
            try:
                frame = camera.read()
            except Exception as exc:
                # 相机读取异常时自动重连
                LOGGER.warning("Camera read failed: %s. Reconnecting...", exc)
                camera.close()
                camera.connect()
                continue

            # 处理相机尚未返回帧的情况（可能在启动中）
            if frame is None:
                no_frame_count += 1
                now_ts = time.time()
                # 每 2 秒输出一次告警，避免日志刷屏
                if now_ts - no_frame_last_log_ts > 2.0:
                    LOGGER.warning(
                        "No camera frame yet (count=%d). Check camera stream or timeout setting.",
                        no_frame_count,
                    )
                    no_frame_last_log_ts = now_ts
                continue

            no_frame_count = 0
            # 解包相机帧：
            #   rgb:           RGB 图像 (H,W,3) uint8
            #   depth_rgb:     深度伪彩色图 (H,W,3) uint8（用于喂给模型的深度键）
            #   depth_u16:     原始深度图 (H,W) uint16，单位 mm（用于日志/调试）
            #   depth_vis_bgr: 深度可视化 BGR 图像（用于 GUI 显示）
            rgb, depth_rgb, depth_u16, depth_vis_bgr = frame

            # --- Step 2: 读取机器人状态 ---
            if args.dry_run:
                # dry_run 模式：用全零代替真实状态，不连接机器人
                state_full = np.zeros((45,), dtype=np.float32)
            else:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                # 获取 URDF 空间下的 45D 状态向量
                state_full = base.get_robot_state_urdf(client)
            # 裁剪/填充到模型期望的 state_dim 维度
            state_model = base.fit_vector(state_full, state_dim)

            # --- Step 3: 多视觉输入推理（仅在 chunk 用完时触发） ---
            if action_chunk is None or chunk_idx >= len(action_chunk):
                action_chunk = infer_single_action_multivis(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    device=device,
                    task=args.task,
                    robot_type=args.robot_type,
                    visual_keys=visual_keys,
                    state_key=state_key,
                    rgb=rgb,
                    depth_rgb=depth_rgb,
                    state_model=state_model,
                    action_dim=action_dim,
                )
                chunk_idx = 0

            action_model = action_chunk[chunk_idx]
            chunk_idx += 1

            # --- Step 4: 动作后处理 ---
            # 将模型输出的动作向量适配到 GR2 的 35D URDF 动作空间
            # （0:7 左臂, 7:14 右臂, 14:20 左手, 20:26 右手, 26:28 头, 28:29 腰, 29:35 底盘）
            raw_action_urdf = base.fit_vector(action_model, 35)
            prev_action_for_diag = prev_action_urdf
            action_urdf = raw_action_urdf.copy()
            if not args.disable_clamp:
                # 夹紧非手部关节的动作值到安全范围，防止机器人超出关节限位
                action_urdf = base.clamp_non_hand_joint_action(action_urdf)
            if prev_action_urdf is not None:
                # 限制步间动作变化量，使运动更平滑（避免突变/抖动）
                action_urdf = base.stabilize_action(action_urdf, prev_action_urdf, args)

            # --- Step 5: 下发动作到机器人 ---
            if not args.dry_run:
                if client is None:
                    raise RuntimeError("Robot client is not initialized.")
                if first_action and args.transition_time_s > 0:
                    # 首帧：从当前位置平滑过渡到第一个目标动作
                    # 避免机器人突然跳到新位姿
                    base.smooth_transition(client, action_urdf, args.transition_time_s, args.transition_freq)
                    policy.reset()  # 过渡后重置策略缓存，避免残留的 action chunk
                    first_action = False
                else:
                    # 正常帧：直接下发动作
                    base.send_action_to_robot(client, action_urdf, args.send_base)
                    first_action = False

            # --- Step 6: 日志记录 ---
            if step % max(1, args.log_every) == 0:
                # 计算手臂关节动作与当前状态的偏差（用于判断执行效果）
                arm_dim = min(14, state_full.shape[0], action_urdf.shape[0])
                arm_delta = float(np.linalg.norm(action_urdf[:arm_dim] - state_full[:arm_dim]))
                # 提取有效深度范围（排除零值/无效像素）
                valid_depth = depth_u16[depth_u16 > 0]
                depth_range = "empty" if valid_depth.size == 0 else f"{int(valid_depth.min())}..{int(valid_depth.max())}"
                base.log_action_diagnostics(
                    step,
                    arm_delta,
                    depth_range,
                    raw_action_urdf,
                    action_urdf,
                    prev_action_for_diag,
                    args,
                )

            # 记录本帧动作供下一帧平滑使用
            prev_action_urdf = action_urdf.copy()

            # --- Step 7: 可视化与快照 ---
            if gui_enabled:
                # GUI 模式：在 OpenCV 窗口中实时显示 RGB 和深度图
                should_quit, last_vis_ts = base.render_gui_frame(
                    args=args,
                    rgb=rgb,
                    depth_u16=depth_u16,
                    depth_vis_bgr=depth_vis_bgr,
                    last_vis_ts=last_vis_ts,
                    save_dir=save_dir,
                    frame_idx=frame_idx,
                )
                if should_quit:
                    break
            elif periodic_save_every > 0 and step % periodic_save_every == 0:
                # 无 GUI 模式：定期保存快照到磁盘
                base.save_snapshot(
                    save_dir=save_dir,
                    frame_idx=frame_idx,
                    rgb=rgb,
                    depth_vis_bgr=depth_vis_bgr,
                    depth_u16=depth_u16,
                )

            step += 1
            frame_idx += 1
            # 检查是否达到最大步数限制
            if args.max_steps > 0 and step >= args.max_steps:
                LOGGER.info("Reached max steps: %d", args.max_steps)
                break

            # --- 循环频率控制 ---
            elapsed = time.perf_counter() - loop_start
            # 如果本帧耗时超过告警阈值，提示用户优化
            if args.slow_loop_warn_ms > 0 and elapsed * 1000.0 > args.slow_loop_warn_ms:
                now_ts = time.time()
                if now_ts - slow_loop_last_log_ts > 2.0:
                    LOGGER.warning(
                        "Slow control loop: %.1f ms (target %.1f ms). Consider `--no-gui`, `--save-every 0`, or lower `--fps`.",
                        elapsed * 1000.0,
                        1000.0 / max(1e-6, args.fps),
                    )
                    slow_loop_last_log_ts = now_ts
            # sleep 补齐到目标帧率（如 30fps → 每帧 33.3ms）
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")

    # =====================================================================
    # (C) 清理阶段
    # =====================================================================
    finally:
        camera.close()
        if gui_enabled:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        if client is not None:
            client.close()


def main() -> None:
    """脚本入口：配置日志、注册信号处理、解析参数、启动部署。"""
    # 配置全局日志格式
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    def _handle_stop(signum, frame):
        """处理 SIGINT/SIGTERM 信号，将其转换为 KeyboardInterrupt 以触发优雅退出。"""
        del frame
        global _RECEIVED_STOP_SIGNAL
        _RECEIVED_STOP_SIGNAL = True
        LOGGER.warning("Received signal %s, exiting cleanly.", signum)
        raise KeyboardInterrupt

    # 注册信号处理器
    signal.signal(signal.SIGINT, _handle_stop)    # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_stop)   # kill / systemd stop
    signal.signal(signal.SIGTSTP, signal.SIG_IGN) # 忽略 Ctrl+Z，防止意外挂起导致机器人失控

    # 解析命令行参数（复用 base 脚本的 argparse 定义）
    args = base.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        if _RECEIVED_STOP_SIGNAL:
            LOGGER.info("Stopped by signal.")
        else:
            LOGGER.info("Interrupted by user.")


if __name__ == "__main__":
    main()
