#!/usr/bin/env python3

"""Client helpers for the dual-model GR2 PI0 RGB wrist inference server."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

DEFAULT_UNIX_SOCKET_PATH = "/tmp/gr2_dual_pi0_rgb_wrist.sock"


def send_request(
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    request = {"method": method, "payload": payload or {}}
    socket_path = str(Path(unix_socket_path).expanduser())

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))

        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break

    if not chunks:
        raise RuntimeError("server returned an empty response")

    raw = b"".join(chunks).decode("utf-8").strip()
    response = json.loads(raw)
    if not isinstance(response, dict) or "code" not in response or "data" not in response:
        raise RuntimeError(f"unexpected response payload: {response}")
    return response


def request_ok(
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    response = send_request(
        method,
        payload,
        unix_socket_path=unix_socket_path,
        timeout_s=timeout_s,
    )
    code = int(response["code"])
    data = response["data"]
    if code >= 400:
        raise RuntimeError(f"request failed with code={code}: {data}")
    return data


def start_green_to_yellow(
    *,
    max_steps: int | None = None,
    fps: float | None = None,
    fsm_state: int | None = None,
    stop_timeout_s: float | None = None,
    restart: bool = False,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"restart": restart}
    if max_steps is not None:
        payload["max_steps"] = max_steps
    if fps is not None:
        payload["fps"] = fps
    if fsm_state is not None:
        payload["fsm_state"] = fsm_state
    if stop_timeout_s is not None:
        payload["stop_timeout_s"] = stop_timeout_s
    return request_ok(
        "start_green_to_yellow",
        payload,
        unix_socket_path=unix_socket_path,
        timeout_s=timeout_s,
    )


def start_yellow_to_green(
    *,
    max_steps: int | None = None,
    fps: float | None = None,
    fsm_state: int | None = None,
    stop_timeout_s: float | None = None,
    restart: bool = False,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"restart": restart}
    if max_steps is not None:
        payload["max_steps"] = max_steps
    if fps is not None:
        payload["fps"] = fps
    if fsm_state is not None:
        payload["fsm_state"] = fsm_state
    if stop_timeout_s is not None:
        payload["stop_timeout_s"] = stop_timeout_s
    return request_ok(
        "start_yellow_to_green",
        payload,
        unix_socket_path=unix_socket_path,
        timeout_s=timeout_s,
    )


def stop(
    *,
    timeout_s: float = 30.0,
    wait_timeout_s: float = 5.0,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
) -> dict[str, Any]:
    return request_ok(
        "stop",
        {"timeout_s": wait_timeout_s},
        unix_socket_path=unix_socket_path,
        timeout_s=timeout_s,
    )


def status(
    *,
    timeout_s: float = 30.0,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
) -> dict[str, Any]:
    return request_ok("status", unix_socket_path=unix_socket_path, timeout_s=timeout_s)


def health(
    *,
    timeout_s: float = 30.0,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
) -> dict[str, Any]:
    return request_ok("health", unix_socket_path=unix_socket_path, timeout_s=timeout_s)


def set_pd(
    *,
    model: str | None = None,
    kp: dict[str, list[int | float]] | None = None,
    kd: dict[str, list[int | float]] | None = None,
    apply_now: bool = True,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"apply": apply_now}
    if model:
        payload["model"] = model
    if kp:
        payload["kp"] = kp
    if kd:
        payload["kd"] = kd
    return request_ok("set_pd", payload, unix_socket_path=unix_socket_path, timeout_s=timeout_s)


def get_pd(
    *,
    model: str | None = None,
    unix_socket_path: str = DEFAULT_UNIX_SOCKET_PATH,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if model:
        payload["model"] = model
    return request_ok("get_pd", payload, unix_socket_path=unix_socket_path, timeout_s=timeout_s)


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client for the dual-model GR2 PI0 RGB wrist server")
    parser.add_argument(
        "--unix-socket-path",
        type=str,
        default=DEFAULT_UNIX_SOCKET_PATH,
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["start-green-to-yellow", "start-yellow-to-green"]:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--max-steps", type=int, default=None)
        subparser.add_argument("--fps", type=float, default=None)
        subparser.add_argument("--fsm-state", type=int, default=None)
        subparser.add_argument("--stop-timeout-s", type=float, default=None)
        subparser.add_argument("--restart", action="store_true")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--wait-timeout-s", type=float, default=5.0)

    subparsers.add_parser("status")
    subparsers.add_parser("health")

    # set-pd: 热更新 PD 增益
    set_pd_parser = subparsers.add_parser("set-pd", help="热更新 PD 增益（不重启 server）")
    set_pd_parser.add_argument("--model", type=str, default=None,
                               help="更新哪个模型的预设 (green_to_yellow / yellow_to_green)，不传则只下发不存预设")
    set_pd_parser.add_argument("--kp", type=str, default=None,
                               help='JSON 格式 kp，如 \'{"right_manipulator": [270, 250, 95, 95, 45, 45, 45]}\'')
    set_pd_parser.add_argument("--kd", type=str, default=None,
                               help='JSON 格式 kd，如 \'{"right_manipulator": [10, 10, 5, 5, 5, 5, 5]}\'')
    set_pd_parser.add_argument("--no-apply", action="store_true",
                               help="只更新预设，不立即下发到机器人")

    # get-pd: 查看当前 PD 增益
    get_pd_parser = subparsers.add_parser("get-pd", help="查看当前 PD 增益预设")
    get_pd_parser.add_argument("--model", type=str, default=None,
                               help="查看指定模型的 PD (green_to_yellow / yellow_to_green)，不传则查看全部")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "start-green-to-yellow":
        _print_json(
            start_green_to_yellow(
                max_steps=args.max_steps,
                fps=args.fps,
                fsm_state=args.fsm_state,
                stop_timeout_s=args.stop_timeout_s,
                restart=args.restart,
                unix_socket_path=args.unix_socket_path,
                timeout_s=args.timeout_s,
            )
        )
        return

    if args.command == "start-yellow-to-green":
        _print_json(
            start_yellow_to_green(
                max_steps=args.max_steps,
                fps=args.fps,
                fsm_state=args.fsm_state,
                stop_timeout_s=args.stop_timeout_s,
                restart=args.restart,
                unix_socket_path=args.unix_socket_path,
                timeout_s=args.timeout_s,
            )
        )
        return

    if args.command == "stop":
        _print_json(
            stop(
                timeout_s=args.timeout_s,
                wait_timeout_s=args.wait_timeout_s,
                unix_socket_path=args.unix_socket_path,
            )
        )
        return

    if args.command == "status":
        _print_json(status(timeout_s=args.timeout_s, unix_socket_path=args.unix_socket_path))
        return

    if args.command == "health":
        _print_json(health(timeout_s=args.timeout_s, unix_socket_path=args.unix_socket_path))
        return

    if args.command == "set-pd":
        kp = json.loads(args.kp) if args.kp else None
        kd = json.loads(args.kd) if args.kd else None
        _print_json(
            set_pd(
                model=args.model,
                kp=kp,
                kd=kd,
                apply_now=not args.no_apply,
                unix_socket_path=args.unix_socket_path,
                timeout_s=args.timeout_s,
            )
        )
        return

    if args.command == "get-pd":
        _print_json(
            get_pd(
                model=args.model,
                unix_socket_path=args.unix_socket_path,
                timeout_s=args.timeout_s,
            )
        )
        return

    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
