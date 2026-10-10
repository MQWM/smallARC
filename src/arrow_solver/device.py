from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess

import cv2
import numpy as np


class DeviceError(RuntimeError):
    pass


@dataclass
class DeviceInfo:
    serial: str
    state: str
    description: str


def tool_path(name: str) -> str:
    return shutil.which(name) or f"/opt/homebrew/bin/{name}"


class AndroidDevice:
    def __init__(self, serial: str, adb: str | None = None):
        if not serial:
            raise DeviceError("请明确选择设备")
        self.serial = serial
        self.adb = adb or tool_path("adb")

    def command(self, *args: str, timeout: float = 10) -> bytes:
        try:
            result = subprocess.run([self.adb, "-s", self.serial, *args], capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(f"ADB 不可用或超时：{exc}") from exc
        if result.returncode:
            raise DeviceError(result.stderr.decode(errors="replace").strip() or "ADB 命令失败")
        return result.stdout

    def screenshot(self):
        data = self.command("exec-out", "screencap", "-p")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise DeviceError("ADB 未返回有效的 PNG 截图")
        return image

    def tap(self, x: int, y: int):
        self.command("shell", "input", "tap", str(int(x)), str(int(y)), timeout=5)


def list_devices(adb: str | None = None) -> list[DeviceInfo]:
    try:
        result = subprocess.run([adb or tool_path("adb"), "devices", "-l"],
                                capture_output=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceError(f"无法列出设备：{exc}") from exc
    if result.returncode:
        raise DeviceError(result.stderr.decode(errors="replace"))
    devices = []
    for line in result.stdout.decode(errors="replace").splitlines()[1:]:
        fields = line.split(maxsplit=2)
        if len(fields) >= 2:
            devices.append(DeviceInfo(fields[0], fields[1], fields[2] if len(fields) > 2 else ""))
    return devices


class Mirror:
    def __init__(self):
        self.process = None
        self.serial = None

    def start(self, serial: str, scrcpy: str | None = None):
        if self.process and self.process.poll() is None:
            if self.serial == serial:
                return
            self.close()
        try:
            self.process = subprocess.Popen([scrcpy or tool_path("scrcpy"), "-s", serial,
                                             "--window-title", "小小箭头 · 手机画面"],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.serial = serial
        except OSError as exc:
            raise DeviceError(f"scrcpy 启动失败：{exc}") from exc

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.process = None
        self.serial = None

