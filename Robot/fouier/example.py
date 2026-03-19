# -*- coding: utf-8 -*-
"""
GR2 机器人控制类使用示例
"""
import time
from gr2_robot import GR2Robot

def main():
    print("==========================================")
    print("      GR2 Robot SDK 封装类测试示例")
    print("==========================================")

    # 1. 初始化机器人
    # 请确保 Docker 容器中的 AuroraCore 正在运行
    # 这里的 domain_id 必须与服务端配置一致
    try:
        robot = GR2Robot(domain_id=123, robot_name="gr2t2v2")
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    try:
        # 2. 切换到站立模式
        # 许多关节控制只有在 PdStand 模式下才生效
        input("按回车键切换到站立模式 (PD Stand)...")
        robot.stand()
        print("机器人已站立。")

        # 3. 头部运动示例
        input("按回车键测试头部运动 (点头/摇头)...")
        # 头部关节: [俯仰(Pitch), 偏航(Yaw)]
        # 抬头
        robot.move_single_group(GR2Robot.GROUP_HEAD, [-0.3, 0.0], duration=1.0)
        # 低头
        robot.move_single_group(GR2Robot.GROUP_HEAD, [0.3, 0.0], duration=1.0)
        # 回正
        robot.move_single_group(GR2Robot.GROUP_HEAD, [0.0, 0.0], duration=1.0)

        # 4. 手臂运动示例 (调用封装好的挥手动作)
        input("按回车键测试左手挥手动作...")
        robot.wave_hand(side="left")

        # 5. 自定义多关节协同运动
        input("按回车键测试多关节协同 (双手抱头)...")
        # 定义目标姿态
        target_pose = {
            # 左臂: 抬起并弯曲
            GR2Robot.GROUP_LEFT_ARM: [-0.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
            # 右臂: 对称动作
            GR2Robot.GROUP_RIGHT_ARM: [-0.5, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0],
            # 头部: 稍稍抬头
            GR2Robot.GROUP_HEAD: [-0.2, 0.0]
        }
        robot.move_joints(target_pose, duration=2.0)

        # 保持一会儿
        time.sleep(1.0)

        # 复位
        print("正在复位...")
        robot.reset_upper_body(duration=2.0)

        # 6. 行走控制示例 (仅做演示，需在开阔场地测试)
        print("\n注意: 下一步将测试行走控制指令。")
        print("为安全起见，本次测试仅发送 0 速度指令，确保链路正常。")
        input("按回车键测试行走模式切换...")

        robot.walk_mode()
        # 开启摆臂
        robot.enable_arm_sway(True)

        # 发送速度指令 (这里设为0，实际使用时可修改参数)
        # robot.set_velocity(0.1, 0, 0) # 前进 0.1 m/s
        for _ in range(50): # 发送一段时间指令
            robot.set_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)

        robot.set_velocity(0, 0, 0) # 停止
        robot.enable_arm_sway(False)
        print("行走测试结束。")

        # 7. 结束前的准备
        input("按回车键结束程序并复位...")
        robot.stand() # 切回站立模式
        robot.reset_upper_body()

    except KeyboardInterrupt:
        print("\n检测到键盘中断，正在停止...")
    finally:
        # 关闭连接
        robot.close()
        print("程序已退出。")

if __name__ == "__main__":
    main()
