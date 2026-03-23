[33mcommit dbf7336553c1a4f83af3850083bbc2a77258e3cd[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mlerobot-gr2[m[33m)[m
Author: heliang pu <2303935680@qq.com>
Date:   Thu Mar 19 16:24:32 2026 +0800

    feat(inference): 双模型推理服务支持 per-task PD 增益及运行时热更新
    
    - 为 take_out / put_in 任务分别定义独立的 PD 增益预设
    - 切换模型时自动下发对应的 PD 参数
    - 新增 set_pd / get_pd API，支持运行时热更新 PD 增益无需重启
    - client 新增 set-pd / get-pd 子命令
    - 新增 inference 目录 README 文档

[33mcommit 4029afb2ab9088108659f77013c4b7445f671da7[m
Author: heliang pu <2303935680@qq.com>
Date:   Tue Mar 17 16:17:29 2026 +0800

    fix(scripts): tune gr2 rgb wrist right arm gains

[33mcommit 52eac90ce9379c7d0120696f1a23472b1950a8d5[m[33m ([m[1;31morigin/lerobot-gr2[m[33m)[m
Author: heliang pu <2303935680@qq.com>
Date:   Tue Mar 17 14:15:02 2026 +0800

    chore(train): remove obsolete train scripts

[33mcommit b15be8e7e4245be78ef0c955b2f44f3f248e4581[m
Author: heliang pu <2303935680@qq.com>
Date:   Tue Mar 17 14:07:43 2026 +0800

    feat(scripts): add gr2 pi0 training launcher

[33mcommit 9855d86ffadbe2d92a570681e2a4893a38c01dcb[m
Author: heliang pu <2303935680@qq.com>
Date:   Tue Mar 17 13:32:02 2026 +0800

    feat(scripts): 新增 GR2 PI0 RGB 腕部相机部署脚本

[33mcommit 79a677ca57d825bab1b846f3f599e0d09ad6396d[m
Author: heliang pu <2303935680@qq.com>
Date:   Tue Mar 17 13:11:13 2026 +0800

    feat(scripts): 新增 GR-2 数据集回放脚本及使用文档
    
    支持从 LeRobot v3.0 数据集回放 episode 到 GR-2 机器人，
    包含关节限位、动作平滑、首帧过渡等安全机制。

[33mcommit c3356cd0854787b0ab56675bed7bdffe600f901e[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Mar 13 17:44:48 2026 +0800

    chore: 移除 projects/ 目录及无关文件，保持仓库专注于 LeRobot GR-2

[33mcommit 3f5ee6c353233b6c6b58dcf15cd8f7878e8250f9[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Mar 13 17:32:37 2026 +0800

    docs: 重写项目 README，匹配 GR-2 实际工具链内容

[33mcommit 2d39fcdae21201bfb89ad4b26a27f7c1e0297c31[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Mar 13 17:20:23 2026 +0800

    feat(scripts): 新增交互式 episode 筛选工具和文档
    
    - 添加 review_episodes.py: 基于 Rerun GUI 逐个回放 episode，支持方向键导航和标记删除
    - 更新 scripts/README.md: 添加数据集清洗工具使用说明

[33mcommit d0cd9db4e955ae0de96f1456b8b4f50d5b3a4b61[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Mar 6 18:03:24 2026 +0800

    feat(scripts): 新增 GR2 PI0 推理服务和训练恢复功能
    
    - 新增 gr2_pi0_inference_service.py：Unix Socket 推理服务，用于 RoboOS 对接
    - 新增 start_gr2_pi0_inference_service.sh：推理服务启动脚本
    - 新增 resume_pi0_gr2_pick_to_10am.sh：恢复训练脚本
    - 更新 train_pi0_gr2_pick.sh：支持从 checkpoint 恢复训练
    - 更新 README.md：添加推理服务使用文档

[33mcommit 3560b8a949cf43ecae305754f77fe47b15b82a80[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Mar 6 10:21:05 2026 +0800

    fix(scripts): tune GR2 PI0 deploy params and add action diagnostics

[33mcommit d7f1fcc714d841c9acf513688adb9431d9975ae3[m
Author: heliang pu <2303935680@qq.com>
Date:   Thu Mar 5 15:46:42 2026 +0800

    feat(scripts): switch GR2 PI0 deploy to rgbd entrypoint and add model-output debug tool

[33mcommit a0a6d59fb95cc22a10760cfd39512311293a0b53[m
Author: heliang pu <2303935680@qq.com>
Date:   Thu Mar 5 13:41:17 2026 +0800

    fix(scripts): 修复 FSM 默认状态为 11(upper_body_cmd)，新增多视觉部署脚本和工具

[33mcommit 4e830400ae6dbffcdffe95a37b6f8cd720f93bae[m
Author: heliang pu <2303935680@qq.com>
Date:   Thu Mar 5 10:53:10 2026 +0800

    docs: 添加 GR2 关节对齐中文文档，注释多视觉部署脚本
    
    - 重写 fourier_gr2_joint_alignment.mdx 为中文详细版，涵盖 45D 状态/35D 动作
      维度映射、手部 SDK↔URDF 转换公式、关节安全限位、动作平滑机制、部署检查清单
    - 为 deploy_gr2_pi0_multivis.py 添加完整中文注释

[33mcommit 64a6c13c0701c049c8aafb2d4207ebc2b962da8c[m[33m ([m[1;32mmain[m[33m)[m
Author: heliang pu <2303935680@qq.com>
Date:   Wed Mar 4 17:30:25 2026 +0800

    feat(scripts): add Pi0 GR2 pick training script
    
    Training config for Pi0 policy on GR2 pick dataset with 45D state,
    35D action, bfloat16, gradient checkpointing, and local dataset support.

[33mcommit 2108809c4b75274fdb88e59debacb37bfedfed64[m
Author: heliang pu <2303935680@qq.com>
Date:   Wed Mar 4 13:56:02 2026 +0800

    refactor(scripts): rewrite gr2 pi0 deploy with Aurora SDK + Orbbec camera
    
    Complete rewrite of the GR2 Pi0 deployment script with proper
    processor pipeline integration, 45D state / 35D action mapping,
    and Orbbec RGB-D camera support via OpenCV.

[33mcommit 7e3f0383a481605db11489c19c51f52468a66834[m
Author: heliang pu <2303935680@qq.com>
Date:   Mon Mar 2 15:41:14 2026 +0800

    fix(scripts): stabilize gr2 pi0 deploy hand mapping and camera handling

[33mcommit 563f22373d604b3ce2c64465b45d20d1c6e75c9a[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Feb 27 15:18:34 2026 +0800

    fix: 修复 GR2 部署脚本维度映射，新增 PI0 部署脚本
    
    deploy_gr2_act.py:
    - 修复 waist 维度: 3D → 1D (GR-2 只有 waist_yaw)，action 37D → 35D
    - 修复 state 读取顺序: 对齐 GR2_JOINT_ORDER (left_hand 在 head 前面)
    - 移除错误的 HAND_ACTION_TO_SDK 重排 (数据集已是 SDK 顺序)
    
    deploy_gr2_pi0.py (新增):
    - PI0 策略部署，维度与 ACT 对齐 (action 35D, state 45D)
    - 支持 --task 参数 (language prompt)
    - 模型预热 + Flow Matching denoising
    - action[:35] 截断处理 (PI0 pad 到 max_action_dim)

[33mcommit 470a481189a6dce284694e2d006ef3cade7730f2[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Feb 27 13:20:14 2026 +0800

    feat: lerobot with GR2 customizations and merged scripts
    
    - LeRobot base from huggingface/lerobot
    - Fourier GR2 robot wrapper and docs
    - Merged scripts from lerobot-fmc3 (training guide, pi0 train script)
    - Custom train/inference scripts for ACT and PI0
    - deploy_gr2_act.py for GR2 deployment

[33mcommit f971b46a4023634f1d3a0f51868192b498146105[m[33m ([m[1;31morigin/main[m[33m)[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Feb 27 14:18:44 2026 +0800

    docs: 更新 README，添加完整的项目文档
    
    包含项目结构、各子项目说明、快速开始指南、系统架构、
    数据集维度映射、通信协议及环境配置等详细信息。

[33mcommit 16cba362b93256feac0e17542e34198d61ea45d7[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Feb 13 16:03:54 2026 +0800

    heliangp:脚本修正去除噪声，增加结构说明和适配说明

[33mcommit b27f6e76e5e9c9d334c0a14518be1869f9963ee1[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Feb 13 13:14:38 2026 +0800

    heliang：增加了parqut-->lerobot数据集格式转换脚本

[33mcommit 0e6f6233264d5bdbd5f6968570fa88ba135b7a7c[m
Author: heliang pu <2303935680@qq.com>
Date:   Fri Jan 30 16:09:28 2026 +0800

    chore: 规范化 gitignore 规则，清理自动生成文件

[33mcommit 775c2521f43f7df67cf36eecbfe19a73e0ff28df[m
Author: haoanw <wanghaoan.victor@gmail.com>
Date:   Thu Jan 29 15:46:53 2026 +0100

    haoanw: delete automatica generated files

[33mcommit 1c2244624987b528cd4c689f7832ec47e1d6b7ff[m
Author: heliang pu <2303935680@qq.com>
Date:   Thu Jan 29 17:34:20 2026 +0800

    heliangp:修改了路径

[33mcommit 2fac4c694fe0d10a19e5dd9e454af398d395e0ee[m
Author: heliang pu <2303935680@qq.com>
Date:   Thu Jan 29 16:36:20 2026 +0800

    heliangp:迁移

[33mcommit b2932c9cfe2b29cca84e00089743961963cf626b[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Sun Jan 25 11:45:25 2026 +0100

    haoanw: add fourier gr2 and update lerobot scripts

[33mcommit 781f9f0f4f2df816b6bc11ab51872c1a6d5fc7f4[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Sat Jan 24 14:11:54 2026 +0100

    haoanw: update fmc3 adaptions for RoboOS framework

[33mcommit 0665384cf534d6d1b5561d0b621094810b17576e[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Sat Jan 24 13:29:05 2026 +0100

    haoanw: re-upload code

[33mcommit 68053a9a581a63b55e78ee388f308fa4e2b4cb61[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Sat Jan 24 13:19:46 2026 +0100

    haoanw: add fmc3 adaptions for BAAI RoboOS frame-work

[33mcommit be85d58b470892a54725388ff43b349520cb0d14[m
Author: haoanw <wanghaoan.victor@gmail.com>
Date:   Fri Dec 26 16:24:06 2025 +0100

    haoanw: add the agent implementation the project, test later

[33mcommit d8c631fb75344b25cbfb2159af1ac829bbbcadb3[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Wed Dec 24 16:40:44 2025 +0100

    haoanw: init commit for lerobot_robobrain integration

[33mcommit 3a4b5953ece345bd0a83c6116ce86fdd2446a800[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Sat Nov 8 17:32:09 2025 +0100

    haoanw: add evaluate policy script

[33mcommit 6689ad6c9c3aa34cd8d72b0508be913830a35ace[m
Author: Vicc <wanghaoan.victor@gmail.com>
Date:   Fri Nov 7 21:25:20 2025 +0100

    Fix quote formatting in README

[33mcommit 4fe14837592622df7c778ae3735b019f8ac60a92[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Fri Nov 7 17:35:30 2025 +0100

    haoanw: add action model training script

[33mcommit 2026ed2095a91519fd049bbfb2814314b855ca8b[m
Author: haoanw <wanghaoan.victor@gmail.com>
Date:   Wed Nov 5 00:37:17 2025 +0100

    HaWa: add training, recording and reply scripts

[33mcommit e070f37dc88463addd7e28a39d0b5d9e0e518869[m
Author: Vicc <wanghaoan.victor@gmail.com>
Date:   Sat Nov 1 15:43:07 2025 +0100

    haoanw: Add subtitle and quote to README

[33mcommit 2f44d9a84d7f74a341b2f071d60fbf64ba4ebdd5[m
Author: Haoan Wang <wanghaoan.victor@gmail.com>
Date:   Sat Nov 1 15:12:58 2025 +0100

    haoanw: add shell scripts for demo

[33mcommit b6b92cf20b6e6979962290d3832f16810a961a8d[m
Author: haoanw_barney <wanghaoan.victor@gmail.com>
Date:   Fri Oct 24 00:09:23 2025 +0200

    haoanw: add quote from alan kay

[33mcommit 0588756e6569683962990c2923c372770c90e9c1[m
Author: Vicc <wanghaoan.victor@gmail.com>
Date:   Wed Oct 1 22:00:30 2025 +0200

    Initial commit
