# GR2 双模型 PI0 RGB 腕部推理服务

基于 Unix Socket 的双模型推理服务，预加载 take_out / put_in 两个 PI0 checkpoint，运行时通过 client 命令切换，无需重启。

## 快速开始

### 1. 启动 Server

```bash
# 方式一：直接启动
HF_HUB_OFFLINE=1 python scripts/inference/gr2_dual_pi0_rgb_wrist_inference_server.py \
  --device cuda \
  --no-move-to-init-pose-on-start \
  --max-arm-delta 0.03 \
  --arm-ema-alpha 0.2


# 方式二：通过启动脚本（推荐，自动清理旧进程和 socket）
bash scripts/inference/start_gr2_dual_pi0_rgb_wrist_inference_server.sh
```

启动脚本支持环境变量覆盖，常用的：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONDA_ENV` | `lerobot-pi0` | conda 环境名 |
| `TAKE_OUT_CHECKPOINT_PATH` | 见脚本 | take_out 模型路径 |
| `PUT_IN_CHECKPOINT_PATH` | 见脚本 | put_in 模型路径 |
| `FPS` | `15` | 推理帧率 |
| `FSM_STATE` | `11` | Aurora FSM 状态（11=上身控制） |
| `DRY_RUN` | `0` | 设为 1 不连接机器人 |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |

### 2. 控制推理

```bash
CLIENT="python scripts/inference/gr2_dual_pi0_rgb_wrist_client.py"

# 启动 take_out 任务
$CLIENT start-take-out --fps 15 --max-steps 0

# 切换到 put_in 任务（自动停止当前任务再启动）
$CLIENT start-put-in --fps 15 --max-steps 0

# 用不同参数重启同一个任务
$CLIENT start-take-out --fps 30 --max-steps 500 --restart

# 停止当前任务
$CLIENT stop

# 查看状态
$CLIENT status
$CLIENT health
```

### 3. 运行时调整 PD 增益

不需要重启 server，直接热更新：

```bash
# 查看所有模型的 PD 预设
$CLIENT get-pd

# 查看某个模型的 PD
$CLIENT get-pd --model take_out

# 更新 right_manipulator 的 kp 并立即下发
$CLIENT set-pd --model take_out \
  --kp '{"right_manipulator": [280, 220, 95, 70, 50, 50, 50]}'

# 同时改 kp 和 kd
$CLIENT set-pd --model put_in \
  --kp '{"right_manipulator": [250, 200, 80, 55, 40, 40, 40]}' \
  --kd '{"right_manipulator": [8, 8, 4, 4, 4, 4, 4]}'

# 只存预设不立即下发（下次切换到该模型时自动生效）
$CLIENT set-pd --model put_in \
  --kp '{"right_manipulator": [260, 210, 80, 60, 45, 45, 45]}' \
  --no-apply
```

可调的关节组：`left_manipulator`(7)、`right_manipulator`(7)、`waist`(1)、`head`(2)。

## Server 常用参数

```
--take-out-checkpoint-path   take_out 模型路径
--put-in-checkpoint-path     put_in 模型路径
--take-out-task              take_out 任务描述（language conditioning）
--put-in-task                put_in 任务描述
--device                     auto / cuda / cpu
--fps                        默认推理帧率
--fsm-state                  Aurora FSM 状态 (11=上身控制, 10=全身)
--domain-id                  Aurora DDS domain ID (默认 123)
--dry-run                    不连接机器人，仅测试推理流程
--no-gui                     无头模式，不显示摄像头画面
--transition-time-s          首次动作平滑过渡时间 (默认 6s)
--switch-transition-time-s   模型切换时的过渡时间 (默认 1.5s)
--action-ema-alpha           动作 EMA 平滑系数 (默认 0.35)
--max-arm-delta              手臂最大单步变化量 (默认 0.06)
--max-hand-delta             手部最大单步变化量 (默认 0.12)
--wrist-left-serial          左腕 RealSense 序列号
--wrist-right-serial         右腕 RealSense 序列号
```

## Client 子命令一览

| 命令 | 说明 |
|------|------|
| `start-take-out` | 启动 take_out 模型推理 |
| `start-put-in` | 启动 put_in 模型推理 |
| `stop` | 停止当前推理 |
| `status` | 查看详细状态 |
| `health` | 健康检查 |
| `set-pd` | 热更新 PD 增益 |
| `get-pd` | 查看 PD 增益预设 |
