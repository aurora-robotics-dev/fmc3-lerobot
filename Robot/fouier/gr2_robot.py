# -*- coding: utf-8 -*-
import time
import math
import logging
from typing import Dict, List, Optional, Union
from fourier_aurora_client import AuroraClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GR2Robot")

class GR2Robot:
    """
    Fourier GR-2 机器人控制封装类

    该类封装了 fourier_aurora_client 的底层调用，提供了更易用的接口来控制 GR-2 机器人。
    功能包括：
    - 状态机(FSM)管理：切换站立、行走、急停等状态
    - 运动控制：全向移动（速度控制）
    - 关节控制：支持单关节组或多关节组的平滑插值运动
    - 状态获取：获取关节位置、IMU数据等

    Attributes:
        client (AuroraClient): 底层 Aurora 客户端实例
        robot_name (str): 机器人型号名称
        domain_id (int): DDS 通信域 ID
    """

    # 定义常用 FSM 状态常量
    FSM_DEFAULT = 0           # 默认状态
    FSM_JOINT_STAND = 1       # 关节归零站立 (所有执行器归零)
    FSM_PD_STAND = 2          # PD站立 (可控制关节)
    FSM_WALK = 3              # 行走/运动控制 (User Controller A)
    FSM_SECURITY = 9          # 安全保护/急停
    FSM_USER_CMD = 10         # 用户指令模式 (全关节控制)
    FSM_UPPER_BODY_CMD = 11   # 上半身用户指令模式 (腿部禁用)

    # 定义控制组名称常量
    GROUP_LEFT_LEG = "left_leg"
    GROUP_RIGHT_LEG = "right_leg"
    GROUP_WAIST = "waist"
    GROUP_HEAD = "head"
    GROUP_LEFT_ARM = "left_manipulator"
    GROUP_RIGHT_ARM = "right_manipulator"
    GROUP_LEFT_HAND = "left_hand"
    GROUP_RIGHT_HAND = "right_hand"

    def __init__(self, domain_id: int = 123, robot_name: str = "gr2"):
        """
        初始化 GR2 机器人控制器

        Args:
            domain_id (int): DDS 域 ID，必须与机器人服务端配置一致。默认为 123。
            robot_name (str): 机器人名称。默认为 "gr2"。
        """
        self.domain_id = domain_id
        self.robot_name = robot_name

        logger.info(f"正在连接机器人 (Domain ID: {domain_id}, Name: {robot_name})...")
        try:
            self.client = AuroraClient.get_instance(domain_id=domain_id, robot_name=robot_name)
            # 等待连接建立
            time.sleep(1.0)
            logger.info("机器人连接成功。")
        except Exception as e:
            logger.error(f"连接机器人失败: {e}")
            raise

    def close(self):
        """关闭连接并释放资源"""
        if self.client:
            # 尝试停止运动
            try:
                self.set_velocity(0, 0, 0)
            except:
                pass
            logger.info("正在断开连接...")
            # 调用底层客户端的 close 方法释放 DDS 资源
            self.client.close()  # <--- 已启用
            logger.info("连接已断开。")


    # ==================== 状态机 (FSM) 控制 ====================

    def set_state(self, state_id: int, wait_time: float = 1.0):
        """
        切换机器人 FSM 状态

        Args:
            state_id (int): 目标状态 ID。建议使用类常量 (如 GR2Robot.FSM_PD_STAND)。
            wait_time (float): 切换后的等待时间(秒)，确保状态切换完成。
        """
        state_name = {
            self.FSM_DEFAULT: "Default (默认)",
            self.FSM_JOINT_STAND: "Joint Stand (关节归零)",
            self.FSM_PD_STAND: "PD Stand (站立)",
            self.FSM_WALK: "Walk (行走)",
            self.FSM_SECURITY: "Security (急停)",
            self.FSM_USER_CMD: "User Cmd (全用户控制)",
            self.FSM_UPPER_BODY_CMD: "Upper Body Cmd (上半身控制)"
        }.get(state_id, f"Unknown({state_id})")

        logger.info(f"切换状态至: {state_name}")
        self.client.set_fsm_state(state_id)
        if wait_time > 0:
            time.sleep(wait_time)

    def stand(self):
        """切换到原地站立状态 (PD Stand)，此时可以控制上半身关节"""
        self.set_state(self.FSM_PD_STAND)

    def joint_stand(self):
        """
        切换到关节归零站立状态 (Joint Stand)
        所有执行器移动到零位，用于检查关节是否正常工作及零点校准。
        """
        self.set_state(self.FSM_JOINT_STAND)

    def walk_mode(self):
        """切换到行走控制模式 (User Controller A)"""
        self.set_state(self.FSM_WALK)
        # 切换到行走模式后，通常需要设置速度源为 DDS
        self.client.set_velocity_source(2)
        logger.info("已启用速度控制源 (DDS)")

    def stop(self):
        """触发安全保护/急停状态"""
        self.set_state(self.FSM_SECURITY)

    def user_cmd_mode(self):
        """
        切换到用户指令模式 (User Cmd)
        允许执行外部所有关节的位置控制和配置指令。
        """
        self.set_state(self.FSM_USER_CMD)

    def upper_body_mode(self):
        """
        切换到上半身用户指令模式 (Upper Body User Cmd)
        仅允许控制上半身关节（腰、臂、头），腿部执行器在此状态下被禁用。
        """
        self.set_state(self.FSM_UPPER_BODY_CMD)

    # ==================== 状态获取 (新增) ====================

    def get_system_state(self) -> Dict[str, int]:
        """
        获取系统 FSM 状态信息

        Returns:
            Dict[str, int]: 包含 fsm (全身状态), upper_fsm (上半身状态), velocity_source (速度源)
        """
        return {
            "fsm": self.client.get_fsm_state(),
            "upper_fsm": self.client.get_upper_fsm_state(),
            "velocity_source": self.client.get_velocity_source()
        }

    def get_base_imu(self) -> Dict[str, List[float]]:
        """
        获取基座 IMU 数据

        Returns:
            Dict: 包含 orientation (四元数 xyzw), angular_vel (角速度), linear_vel (线速度), acceleration (加速度)
        """
        return {
            "orientation": self.client.get_base_data("quat_xyzw"),
            "angular_vel": self.client.get_base_data("omega_B"),
            "linear_vel": self.client.get_base_data("vel_B"),
            "acceleration": self.client.get_base_data("acc_B")
        }

    def get_joint_state(self, group_name: str, keys: List[str] = ["position"]) -> Dict[str, List[float]]:
        """
        获取关节详细状态 (位置, 速度, 力矩)

        Args:
            group_name (str): 关节组名称
            keys (List[str]): 需要获取的属性列表, 可选 ["position", "velocity", "effort"]

        Returns:
            Dict[str, List[float]]: 键为属性名, 值为对应的数据列表
        """
        result = {}
        for key in keys:
            try:
                result[key] = self.client.get_group_state(group_name, key)
            except Exception as e:
                logger.warning(f"获取关节组 {group_name} 的 {key} 失败: {e}")
        return result

    def get_end_effector_pose(self, group_name: str) -> List[float]:
        """
        获取末端执行器（如手部）的笛卡尔位姿

        Args:
            group_name (str): 关节组名称

        Returns:
            List[float]: 包含位置和姿态的数据列表
        """
        return self.client.get_cartesian_state(group_name)

    # ==================== 姿态与高级控制 (新增) ====================

    def set_stand_offset(self, z: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        """
        调整站立姿态偏移 (仅在 PdStand 模式有效)

        Args:
            z (float): 高度偏移量 (m)，负值表示下蹲
            pitch (float): 俯仰角偏移量 (rad)，正值低头/前倾
            yaw (float): 偏航角偏移量 (rad)，正值左转
        """
        self.client.set_stand_pose(z, pitch, yaw)

    def set_pd_gains(self, kp: Dict[str, List[float]], kd: Dict[str, List[float]]):
        """
        动态设置电机 PD 参数 (用于刚度控制)

        Args:
            kp (Dict): 比例增益 (刚度)，格式 {"group_name": [kp1, kp2...]}
            kd (Dict): 微分增益 (阻尼)，格式 {"group_name": [kd1, kd2...]}
        """
        self.client.set_motor_cfg(kp, kd)

    def set_control_source(self, source_id: int):
        """
        手动设置速度指令来源

        Args:
            source_id (int): 0=手柄(默认), 1=内部规划, 2=DDS(外部SDK控制)
        """
        self.client.set_velocity_source(source_id)

    # ==================== 运动/行走控制 ====================

    def set_velocity(self, vx: float, vy: float, vyaw: float):
        """
        发送全向移动速度指令 (需在 Walk 模式下)

        Args:
            vx (float): 前后移动速度 (m/s)，正值向前，负值向后。
            vy (float): 左右平移速度 (m/s)，正值向左，负值向右。
            vyaw (float): 旋转角速度 (rad/s)，正值向左转(逆时针)，负值向右转。
        """
        # 简单限制幅度，防止过大指令
        vx = max(-1.0, min(1.0, vx))
        vy = max(-0.5, min(0.5, vy))
        vyaw = max(-1.0, min(1.0, vyaw))

        self.client.set_velocity(vx, vy, vyaw)

    def enable_arm_sway(self, enable: bool = True):
        """
        开启或关闭行走时的自然摆臂

        Args:
            enable (bool): True 开启摆臂，False 关闭。
        """
        # set_upper_fsm_state(1) 通常用于开启摆臂
        state = 1 if enable else 0
        self.client.set_upper_fsm_state(state)
        logger.info(f"{'开启' if enable else '关闭'}行走摆臂")

    # ==================== 关节控制 ====================

    def get_joint_positions(self, group_name: str) -> List[float]:
        """
        获取指定关节组的当前位置

        Args:
            group_name (str): 控制组名称 (如 "left_manipulator", "head")

        Returns:
            List[float]: 关节位置列表 (弧度)
        """
        return self.client.get_group_state(group_name)

    def move_joints(self, target_positions: Dict[str, List[float]], duration: float = 2.0, frequency: int = 100):
        """
        平滑移动多个关节组到目标位置 (使用线性插值)

        该函数会阻塞当前线程直到运动完成。

        Args:
            target_positions (Dict[str, List[float]]): 目标位置字典。
                Key 为控制组名称 (如 "left_manipulator")
                Value 为该组的目标关节角度列表 (弧度)
            duration (float): 运动持续时间 (秒)。
            frequency (int): 插值控制频率 (Hz)，建议 >= 100。
        """
        # 1. 获取当前位置作为初始点
        init_positions = {}
        for group in target_positions.keys():
            try:
                current_pos = self.get_joint_positions(group)
                if not current_pos:
                    logger.warning(f"无法获取关节组 {group} 的当前状态，跳过。")
                    continue

                # 检查关节数量是否匹配
                if len(current_pos) != len(target_positions[group]):
                    logger.error(f"关节组 {group} 维度不匹配: 当前{len(current_pos)}, 目标{len(target_positions[group])}")
                    continue

                init_positions[group] = current_pos
            except Exception as e:
                logger.error(f"读取关节组 {group} 失败: {e}")
                return

        if not init_positions:
            logger.warning("没有有效的关节组需要移动。")
            return

        # 2. 执行插值运动
        steps = int(duration * frequency)
        if steps <= 0: steps = 1
        dt = 1.0 / frequency

        logger.info(f"开始关节运动: {list(init_positions.keys())}, 耗时 {duration}s")

        for step in range(steps + 1):
            start_time = time.time()
            progress = step / steps

            # 组合当前帧的所有关节指令
            current_frame_command = {}

            for group, start_pos in init_positions.items():
                target_pos = target_positions[group]
                # 线性插值: p = start + (target - start) * progress
                interp_pos = [s + (t - s) * progress for s, t in zip(start_pos, target_pos)]
                current_frame_command[group] = interp_pos

            # 发送指令
            self.client.set_joint_positions(current_frame_command)

            # 保持频率
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def move_single_group(self, group_name: str, target_pos: List[float], duration: float = 2.0):
        """
        控制单个关节组运动的快捷方法

        Args:
            group_name (str): 控制组名称
            target_pos (List[float]): 目标位置列表
            duration (float): 持续时间
        """
        self.move_joints({group_name: target_pos}, duration=duration)

    # ==================== 预设动作封装 ====================

    def reset_upper_body(self, duration: float = 2.0):
        """
        将上半身所有关节归零 (复位)
        包含: 手臂、头部、腰部。不包含腿部和手部(保持原样)。
        """
        # 目标全0
        # 注意：需确认各关节0位是否为安全位置。对于 GR2，0位通常是伸直或垂下。
        targets = {
            self.GROUP_LEFT_ARM: [0.0] * 7,
            self.GROUP_RIGHT_ARM: [0.0] * 7,
            self.GROUP_WAIST: [0.0],
            self.GROUP_HEAD: [0.0, 0.0]
        }
        self.move_joints(targets, duration=duration)

    def wave_hand(self, side: str = "left"):
        """
        简单的挥手动作示例

        Args:
            side (str): "left" 或 "right"
        """
        group = self.GROUP_LEFT_ARM if side == "left" else self.GROUP_RIGHT_ARM
        # 举手位置 (根据实际机器人构型调整)
        # 假设: [肩俯仰, 肩滚转, 肩偏航, 肘, 腕偏航, 腕俯仰, 腕滚转]
        lift_pose = [-0.5, 0.5, 0, -1.5, 0, 0, 0] if side == "left" else [-0.5, -0.5, 0, -1.5, 0, 0, 0]

        logger.info(f"{side}手 挥手动作开始")
        self.move_single_group(group, lift_pose, duration=1.5)

        # 挥动两次
        wave_out = list(lift_pose)
        wave_in = list(lift_pose)
        wave_out[2] = 0.5  # 肩偏航向外
        wave_in[2] = -0.5 # 肩偏航向内

        for _ in range(2):
            self.move_single_group(group, wave_out, duration=0.5)
            self.move_single_group(group, wave_in, duration=0.5)

        # 复位
        self.move_single_group(group, [0]*7, duration=1.5)
