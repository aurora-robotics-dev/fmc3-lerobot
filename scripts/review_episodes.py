#!/usr/bin/env python
"""交互式 episode 筛选工具，基于 lerobot-dataset-viz 回放，支持键盘方向键导航。

用法:
    python scripts/review_episodes.py \
        --repo-id fmc3_gr2_grab_bottle_into_box_lerobot_ds \
        --root /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds \
        --start-episode 0

键盘操作:
    →  / n / 回车  - 保留并下一个
    ←  / p         - 上一个 (重新审核)
    ↓  / d         - 标记删除并下一个
    ↑  / r         - 重播当前
    j N            - 跳到第 N 个 episode
    q              - 退出保存
"""

import argparse
import json
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path


def get_key():
    """读取单个按键，支持方向键。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                elif ch3 == "B":
                    return "down"
                elif ch3 == "C":
                    return "right"
                elif ch3 == "D":
                    return "left"
            return "esc"
        elif ch == "\r" or ch == "\n":
            return "enter"
        elif ch == "\x03":
            return "ctrl-c"
        else:
            return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_line():
    """读取一行输入（用于 j N 跳转）。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        buf = ""
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                print()
                return buf
            elif ch == "\x7f":  # backspace
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            else:
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def get_total_episodes(root):
    info_path = Path(root) / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)["total_episodes"]


def start_viz(repo_id, root, episode_index):
    """启动 lerobot-dataset-viz 子进程，返回 Popen 对象。"""
    cmd = [
        "lerobot-dataset-viz",
        f"--repo-id={repo_id}",
        f"--root={root}",
        f"--episode-index={episode_index}",
        "--display-compressed-images=true",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    return proc


def stop_viz(proc):
    """停止 viz 子进程。"""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    parser = argparse.ArgumentParser(description="交互式 episode 筛选工具")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--output", type=str, default="episodes_to_delete.json")
    args = parser.parse_args()

    total_episodes = get_total_episodes(args.root)

    # 加载已有进度
    output_path = Path(args.output)
    results = {}  # ep_index -> "keep" | "delete"
    if output_path.exists():
        with open(output_path) as f:
            saved = json.load(f)
        for ep in saved.get("delete", []):
            results[ep] = "delete"
        for ep in saved.get("keep", []):
            results[ep] = "keep"
        n_keep = sum(1 for v in results.values() if v == "keep")
        n_del = sum(1 for v in results.values() if v == "delete")
        print(f"已加载进度: {n_keep} 保留, {n_del} 删除")

    ep = args.start_episode
    proc = None

    def save_progress():
        delete_list = sorted(k for k, v in results.items() if v == "delete")
        keep_list = sorted(k for k, v in results.items() if v == "keep")
        with open(output_path, "w") as f:
            json.dump({"delete": delete_list, "keep": keep_list}, f, indent=2)

    def show_status():
        n_keep = sum(1 for v in results.values() if v == "keep")
        n_del = sum(1 for v in results.values() if v == "delete")
        tag = ""
        if ep in results:
            tag = f" [已标记: {results[ep]}]"
        print(f"\n--- Episode {ep}/{total_episodes-1} (已审: {len(results)}/{total_episodes}, 删除: {n_del}){tag} ---")
        print("  → 保留下一个  ← 上一个  ↓ 删除  ↑ 重播  j 跳转  q 退出")

    print(f"\n数据集共 {total_episodes} 个 episode")
    print("Rerun GUI 窗口会自动弹出")

    try:
        while 0 <= ep < total_episodes:
            stop_viz(proc)
            show_status()
            print("  加载中...")
            proc = start_viz(args.repo_id, args.root, ep)
            print("  回放已启动")

            while True:
                key = get_key()

                if key in ("right", "n", "enter"):
                    if ep not in results:
                        results[ep] = "keep"
                    print(f"  ✓ 保留 episode {ep}")
                    save_progress()
                    ep += 1
                    break

                elif key in ("down", "d"):
                    results[ep] = "delete"
                    print(f"  ✗ 删除 episode {ep}")
                    save_progress()
                    ep += 1
                    break

                elif key in ("left", "p"):
                    if ep > 0:
                        ep -= 1
                        print(f"  ← 回到 episode {ep}")
                    else:
                        print("  已经是第一个了")
                        continue
                    break

                elif key in ("up", "r"):
                    print("  重播中...")
                    stop_viz(proc)
                    proc = start_viz(args.repo_id, args.root, ep)
                    print("  回放已启动")

                elif key == "j":
                    sys.stdout.write("  跳转到: ")
                    sys.stdout.flush()
                    num_str = read_line()
                    try:
                        target = int(num_str.strip())
                        if 0 <= target < total_episodes:
                            ep = target
                            print(f"  → 跳转到 episode {ep}")
                            break
                        else:
                            print(f"  范围: 0-{total_episodes-1}")
                    except ValueError:
                        print("  请输入数字")

                elif key in ("q", "ctrl-c"):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        pass
    finally:
        stop_viz(proc)
        save_progress()

    n_del = sum(1 for v in results.values() if v == "delete")
    n_keep = sum(1 for v in results.values() if v == "keep")
    delete_list = sorted(k for k, v in results.items() if v == "delete")

    print(f"\n审核完成. 保留: {n_keep}, 删除: {n_del}")
    print(f"结果已保存到: {output_path}")

    if delete_list:
        print(f"\n清洗命令:")
        print(f"  lerobot-edit-dataset delete_episodes \\")
        print(f"      --dataset.repo_id={args.repo_id} \\")
        print(f"      --dataset.root={args.root} \\")
        print(f"      --episode_indices='{json.dumps(delete_list)}' \\")
        print(f"      --output_dir=<cleaned_output_path>")


if __name__ == "__main__":
    main()
