import subprocess
from types import SimpleNamespace

import numpy as np
import cv2
import pytest

from arrow_solver.device import AndroidDevice, DeviceError, list_devices


def test_adb_click_uses_argument_array_and_explicit_serial(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(subprocess, "run", run)
    AndroidDevice("device with spaces", "/custom/adb").tap(50, 80)
    assert calls[0][0] == ["/custom/adb", "-s", "device with spaces", "shell", "input", "tap", "50", "80"]
    assert "shell" not in calls[0][1]
    assert calls[0][1]["timeout"] == 5


def test_screenshot_decodes_binary(monkeypatch):
    image = np.full((50, 30, 3), 135, np.uint8)
    _, png = cv2.imencode(".png", image)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=png.tobytes(), stderr=b""))
    assert np.array_equal(AndroidDevice("test").screenshot(), image)


def test_device_listing_keeps_authorization_state(monkeypatch):
    output = b"List of devices attached\na device product:x\nb unauthorized\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=output, stderr=b""))
    devices = list_devices()
    assert [(d.serial, d.state) for d in devices] == [("a", "device"), ("b", "unauthorized")]


def test_timeout_is_reported(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("adb", 5)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(DeviceError, match="超时"):
        AndroidDevice("test").tap(1, 2)

