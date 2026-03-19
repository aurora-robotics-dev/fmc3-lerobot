# Fourier GR-2 Robot SDK 封装包

这是一个基于 Fourier Aurora SDK 的 Python 高层封装库，旨在简化 GR-2 机器人的开发流程。

## 特性

*   **简单易用**: 将复杂的底层调用封装为直观的 Python 类方法。
*   **自动插值**: 关节控制内置线性插值算法，确保运动平滑，防止机器抖动。
*   **状态管理**: 简单的方法切换站立、行走、急停等状态。
*   **详细注释**: 代码包含完整的中文注释和参数说明。

## 目录结构

```text
Robot/
├── __init__.py      # 导出接口
├── gr2_robot.py     # 核心封装类
├── example.py       # 使用示例代码
├── setup.py         # 安装配置文件
├── requirements.txt # 依赖列表
└── README.md        # 说明文档
```

## 安装方法

### 方法 1: 直接复制 (推荐)
将整个 `Robot` 文件夹复制到你的项目根目录下。

```python
# 在你的代码中引用
from Robot import GR2Robot

robot = GR2Robot()
```

### 方法 2: 作为库安装
进入 `Robot` 目录并使用 pip 安装：

```bash
cd Robot
pip install .
```

之后可以在任何地方引用：
```python
from fourier_gr2_robot import GR2Robot
# 注意: setup.py 中的 name 是 fourier_gr2_robot，但包名取决于文件夹结构
# 如果是扁平结构安装，可能需要根据 setup.py 的配置调整
```

## 快速开始

### 1. 初始化
```python
from Robot import GR2Robot

# 确保 Domain ID 与机器人一致
robot = GR2Robot(domain_id=123, robot_name="gr2t2v2")
```

### 2. 关节控制 (自动插值)
```python
# 切换到站立模式才能控制上半身
robot.stand()

# 定义目标位置 (弧度)
target_positions = {
    "head": [0.5, 0.0],  # [俯仰, 偏航]
    "left_manipulator": [-0.5, 0, 0, -1.0, 0, 0, 0]
}

# 2.0秒内平滑移动到目标
robot.move_joints(target_positions, duration=2.0)
```

### 3. 行走控制
```python
robot.walk_mode()

# 前进 0.2 m/s
robot.set_velocity(vx=0.2, vy=0.0, vyaw=0.0)
```

## 注意事项
1. **安全第一**: 真机调试时请确保急停开关在手边。
2. **插值**: `move_joints` 函数会阻塞当前线程直到运动完成，以保证平滑插值。
