## 资料

SO ARM100/101

教程

视频教程：【从零搭建具身智能机械臂1：组装和调试】 https://www.bilibili.com/video/BV1k6UUBFEw4/?spm_id_from=333.337.search-card.all.click&vd_source=99856683a3741f9af3785ccd90272 英文wiki教程：https://wiki.seeedstudio.com/cn/lerobot_so100m_new/ 中文wiki教程：https://wiki.seeedstudio.com/cn/lerobot_so100m_new/ 规格书： STS3215 Servo Motor Datasheet： https://drive.weixin.qq.com/s?k=AGEAZwfLABEnnGawbMAT8AawY5AOc Power Adapter Datasheet： https://drive.weixin.qq.com/s?k=AGEAZwfLABEowXXaYyAT8AawY5AOc Serial Bus Servo Drive Board DataSheet： https://drive.weixin.qq.com/s?k=AGEAZwfLABE0CUwdz1AT8AawY5AOc https://files.seeedstudio.com/products/NVIDIA/reComputer-J301x-datasheet.pdf https://files.seeedstudio.com/products/NVIDIA/reComputer-J401x-datasheet.pdf SO-ARM100 3D printed parts： https://github.com/TheRobotStudio/SO-ARM100 

上位机： https://gitee.com/ftservo/fddebug/tree/master 如果识别不到请下载用上位机校准舵机的中位。 如果要搭配摄像头使用： ●摄像头使用：配海康1080P定焦摄像头 【淘宝】https://e.tb.cn/h.hwXivbMdbkkpVhm?tk=ywhw4mMK0hs HU071 「海康威视usb摄像头电脑外置带麦克风一体台式笔记本直播网课面试」 点击链接直接打开 或者 淘宝搜索直接打开 ●摄像头支架安装方法： 【淘宝】假一赔四 https://e.tb.cn/h.hEyQO4tVH5Fis2Y?tk=XiSX4mMpqQZ CA381 「【专业俯拍】2024新款手机支架桌面直播三脚架录视频vlog拍美食神器专用书法网课开箱拍摄补光灯拍照设备架子」 点击链接直接打开 或者 淘宝搜索直接打开 （推荐选择加重【桌面伸缩加长款】） https://detail.tmall.com/item.htm?id=827302999782&spm=tbpc.boughtlist.suborder_itemtitle.1.4cde2e8dwbo7GQ&mi_id=00008nrG5OMhhA0_7QsT4rdcSCzm3xuTf8qkz4Urw0Wz94M 摄像头的作用是：图像根据网络模型生成机械臂动作，除了机械臂（从臂）和抓取物体，不要录制其他的东西进去，保证桌面整洁 ●电线断了可以买20cm以上就行 （3pin 26cm的） 【淘宝】7天无理由退货 https://e.tb.cn/h.hENJSrHlLtl34fR?tk=qFCV4mbHrGY CZ028 「三针四针5264耐磨端子线60/OD:1.55全黑黑红白橙双头总线舵机线」 点击链接直接打开 或者 淘宝搜索直接打开 

## 标准

```bash
以后数据和模型的id都用这个
策略_框架_组织_机器人型号_任务名称_版本号_类型(模型/数据集)
pi0_lerobot_fmc3_gr2_grab_box_v2_mdl
pi0_lerobot_fmc3_gr2_grab_box_v2_ds
```



## 校准从臂

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=fmc3_robotics_follower_arm
```

## 校准主臂

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=fmc3_robotics_leader_arm
```

## 启动遥操

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=fmc3_robotics_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=fmc3_robotics_leader_arm
```

## 摄像头

```bash

  ┌──────────────┬─────────────────────────────┬────────────────────┐
  │ OpenCV Index │           设备名            │      判断依据      │
  ├──────────────┼─────────────────────────────┼────────────────────┤
  │ Camera #0    │ 1080P USB Camera（海康）    │ 25fps，USB外接设备 │
  ├──────────────┼─────────────────────────────┼────────────────────┤
  │ Camera #1    │ FaceTime高清相机（Mac内置） │ 30fps，内置摄像头  │
  └──────────────┴─────────────────────────────┴────────────────────┘

  所以在 LeRobot 里使用海康摄像头时，camera_index=0；用 Mac
  内置摄像头时，camera_index=1。
```

## 录制

```bash
一轮（episode）的流程：

  ┌─────────────────────┬────────────────────────┐
  │        按键         │          动作          │
  ├─────────────────────┼────────────────────────┤
  │ 准备好后按 右箭头 → │ 开始录制这一轮         │
  ├─────────────────────┼────────────────────────┤
  │ 操作机械臂完成任务  │ 录制中...              │
  ├─────────────────────┼────────────────────────┤
  │ 按 右箭头 →         │ 保存这一轮，进入下一轮 │
  ├─────────────────────┼────────────────────────┤
  │ 按 左箭头 ←         │ 丢弃这一轮，重新录     │
  ├─────────────────────┼────────────────────────┤
  │ 按 Esc              │ 提前结束所有录制       │
  └─────────────────────┴────────────────────────┘
```

## 推理

原始任务（直接放入盒子）

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
  --robot.id=fmc3_robotics_follower_arm \
  --display_data=true \
  --dataset.repo_id=puheliang/eval_lerobot_fmc3_robotics_grab_box_v2 \
  --dataset.single_task="Pick up the tape and place it in the box" \
  --dataset.episode_time_s=1000 \
  --dataset.push_to_hub=false \
  --policy.path=/home/phl/workspace/mymodels/pi0_phl_grab_box_v2
  --dataset.reset_time_s=0
```

经过中间点后再放入盒子（本地评估）

```bash
# 本地评估目录建议每次先清空，避免 FileExistsError
rm -rf /tmp/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1

# 注意：
# 1) --policy.path 不能被换行拆开（不要把 /checkpoints/ 和 last/pretrained_model 分成两行）
# 2) --dataset.reset_time_s=0 避免无 teleop 时在 reset 阶段刷屏
# 3) --dataset.push_to_hub=false 避免 local/... 触发 Hugging Face 403
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM2 \
  --robot.id=fmc3_robotics_follower_arm \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
  --policy.path=/home/phl/workspace/mymodels/pi0_via_middle_finetune_20260310/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --display_data=true \
  --dataset.repo_id=local/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1 \
  --dataset.root=/tmp/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1 \
  --dataset.single_task="Pick up the tape, go to the middle waypoint first, and only then place it into the box" \
  --dataset.num_episodes=100 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false
```

## 录制配置

双摄像头配置

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=fmc3_robotics_follower_arm \
    --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=fmc3_robotics_leader_arm \
    --display_data=true \
    --dataset.repo_id=puheliang/lerobot_fmc3_grab_box_v2 \
    --dataset.num_episodes=400 \
    --dataset.single_task="Pick up the tape and place it in the box" \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=0 \
    --resume=true
```

经过中间点后再放入盒子

如果上次创建数据集失败，先删掉半成品缓存目录：

```bash
rm -rf /home/phl/.cache/huggingface/lerobot/puheliang/lerobot_fmc3_grab_box_via_middle_v1
```

然后重新开始录制：

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=fmc3_robotics_follower_arm \
    --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=fmc3_robotics_leader_arm \
    --display_data=true \
    --dataset.repo_id=puheliang/lerobot_fmc3_grab_box_via_middlebox_v1 \
    --dataset.num_episodes=600 \
    --dataset.single_task="Pick up the tape, place it in the middle waypoint box (white inside), then place it in the final box (black inside)." \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=90 \
    --dataset.reset_time_s=3 \
    --resume=false
```

录完确认没问题后再上传数据集。
## 回放检查

逐个检查数据集里的每个 episode，主要看 `top` 视角里有没有人的手、身体或明显遮挡进入画面：

```bash
for i in $(seq 0 30); do
  echo "===== episode $i ====="
  lerobot-dataset-viz \
    --repo-id puheliang/lerobot_fmc3_grab_box_via_middle_v1 \
    --episode-index $i \
    --mode local \
    --display-compressed-images=false
  read -p "Press Enter for next episode..."
done
```

如果本地窗口没有正常弹出，就改用网页模式：

```bash
for i in $(seq 0 30); do
  echo "===== episode $i ====="
  lerobot-dataset-viz \
    --repo-id puheliang/lerobot_fmc3_grab_box_via_middle_v1 \
    --episode-index $i \
    --mode distant \
    --web-port 9090 \
    --ws-port 9087 \
    --display-compressed-images=false
  read -p "Open http://localhost:9090 and check episode $i, then press Enter..."
done
```
## 端口号

从臂

```bash
/dev/ttyACM2
```

主臂

```bash
/dev/ttyACM1
```

注意：`/dev/ttyACM0` 是 1080P USB Camera，不是机械臂串口。

## 继续训练

基于已有的 pi0 模型继续微调，让它学习“先放到中间点，再放进盒子”这个更长的任务：

```bash
# 这里不需要单独写 --dataset.single_task。
# 训练时会直接读取数据集里已经保存好的 task 文本。
# 这里也不需要再额外写 --policy.type=pi0，
# 因为 --policy.path 指向的 pretrained_model/config.json 里 type 已经是 pi0。
lerobot-train \
    --dataset.repo_id=puheliang/lerobot_fmc3_grab_box_via_middle_v1 \
    --policy.path=/home/phl/workspace/mymodels/pi0_step30000/030000/pretrained_model \
    --output_dir=/home/phl/workspace/mymodels/pi0_via_middle_finetune_20260310 \
    --job_name=pi0_fmc3_grab_box_via_middle_v1 \
    --policy.repo_id=puheliang/pi0_fmc3_grab_box_via_middle_v1 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --batch_size=8 \
    --steps=15000 \
    --save_freq=2000 \
    --eval_freq=2000 \
    --log_freq=100
```

参数说明：

```bash
# --dataset.repo_id
# 使用当前录好的“经过中间点”数据集。

# --policy.path
# 从你上一次已经训练好的 pi0 checkpoint 继续微调，不从零开始。
# 这个目录里的 config.json 已经写了 type=pi0，所以这次训练还是 pi0。

# task
# 不需要在 lerobot-train 里单独写。
# 训练时会从数据集元数据里读取你录制时保存的 task 文本。

# --output_dir
# 这次训练的新输出目录。不要和旧目录重复，否则会报 output_dir 已存在。

# --policy.repo_id
# 如果后续要上传模型到 Hugging Face，这就是新模型名字。

# --policy.push_to_hub=false
# 先只在本地训练，确认效果后再上传，避免中途训练失败还往线上推。

# --steps=15000
# 对当前这种“旧任务基础上加一个中间点”的增量任务，先用 15000 步作为第一版。
# 如果后面你把数据补到 100+ 或 200+ 条，还可以再继续加步数。
```

如果训练中断，继续训练：

```bash
lerobot-train \
    --config_path=/home/phl/workspace/mymodels/pi0_via_middle_finetune_20260310/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

`resume=true` 是继续当前这一次训练，不是重新开一个新微调任务。

如果训练完成，直接用最新的 `last` checkpoint 做推理：

```bash
# 如果这个临时评估目录已经存在，先删掉，避免报 FileExistsError。
rm -rf /tmp/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=fmc3_robotics_follower_arm \
    --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
    --policy.path=/home/phl/workspace/mymodels/pi0_via_middle_finetune_20260310/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --display_data=true \
    --dataset.repo_id=local/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1 \
    --dataset.root=/tmp/eval_pi0_lerobot_fmc3_gr2_grab_box_via_middle_v1 \
    --dataset.single_task="Pick up the tape and place it in the box" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=90 \
    --dataset.push_to_hub=false
```

如果你只想测试老任务“直接放进盒子”，只改这一行：

```bash
    --dataset.single_task="Pick up the tape and place it in the box" \
```

## 上传到huggingface

```bash
# 先把当前数据集上传到 Hugging Face（README.md 会一起上传）
hf upload puheliang/lerobot_fmc3_grab_box_via_middle_v1     /home/phl/.cache/huggingface/lerobot/puheliang/lerobot_fmc3_grab_box_via_middle_v1     .     --repo-type=dataset     --commit-message="Add via-middle SO101 dataset and dataset card"
```

## 从基础模型开始训练

### grab bottle from box to desk (RGB)

```bash
lerobot-train \
    --dataset.repo_id=local/fmc3_gr2_grab_bottle_from_box_to_desk_rgb \
    --dataset.root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_from_box_to_desk_lerobot_ds_rgb \
    --dataset.streaming=false \
    --dataset.video_backend=torchcodec \
    --policy.type=pi0 \
    --policy.pretrained_path=/home/phl/workspace/models/pi0 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.max_state_dim=45 \
    --policy.max_action_dim=35 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=/home/phl/workspace/mymodels/gr2/pi0_gr2_grab_bottle_from_box_to_desk_rgb \
    --job_name=pi0_gr2_grab_bottle_from_box_to_desk_rgb \
    --steps=50000 \
    --save_freq=10000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=Lerobot_Phl_Project \
    --batch_size=8
```

### grab bottle into box (3-cam RGB)

```bash
lerobot-train \
    --dataset.repo_id=local/fmc3_gr2_grab_bottle_into_box_lerobot_ds \
    --dataset.root=/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds \
    --dataset.streaming=false \
    --dataset.video_backend=torchcodec \
    --policy.type=pi0 \
    --policy.pretrained_path=/home/phl/workspace/models/pi0 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.max_state_dim=45 \
    --policy.max_action_dim=35 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=/home/phl/workspace/mymodels/gr2/pi0_gr2_grab_bottle_into_box_rgb \
    --job_name=pi0_gr2_grab_bottle_into_box_rgb \
    --steps=100000 \
    --save_freq=20000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=Lerobot_Phl_Project \
    --batch_size=8
```
