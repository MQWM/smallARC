import cv2
import numpy as np
import pytest

from image_assertions import ScreenVerifier
from arrow_solver.executor import Executor, ExecutionControl, ExecutionError, Cancelled
from arrow_solver.models import Arrow, Board
from arrow_solver.solver import solve


def scene():
    image = np.full((180, 240, 3), (65, 55, 50), np.uint8)
    arrows = [Arrow(1, [(1, 1), (2, 1)], "RIGHT", (100, 60)),
              Arrow(2, [(4, 1), (5, 1)], "RIGHT", (220, 60))]
    for a in arrows:
        cv2.arrowedLine(image, (20 + a.cells[0][0] * 40, 60), a.click_point, (80, 220, 210), 8, tipLength=.3)
    b = Board(240, 180, (0, 0, 240, 180), (20, 20), (40, 40), 6, 4, arrows, line_width=8)
    return image, b


class FakeDevice:
    serial = "test-only"

    def __init__(self, image, fail=False):
        self.image = image.copy()
        self.taps = []
        self.fail = fail

    def screenshot(self):
        return self.image.copy()

    def tap(self, x, y):
        self.taps.append((x, y))
        if not self.fail:
            # Entire connected instance disappears after this fake tap.
            cv2.rectangle(self.image, (x - 48, y - 20), (min(x + 15, 239), y + 20), (65, 55, 50), -1)


def test_cancel_before_start(tmp_path):
    source, b = scene()
    d = FakeDevice(source)
    control = ExecutionControl()
    control.cancel.set()
    ex = Executor(d, b, solve(b), source, tmp_path, control)
    with pytest.raises(Cancelled):
        ex.run()
    assert not d.taps


def test_stop_during_sent_step_never_clicks_next(tmp_path):
    source, b = scene()
    d = FakeDevice(source)
    control = ExecutionControl()
    def report(kind, data):
        if kind == "sent":
            control.cancel.set()
    ex = Executor(d, b, solve(b), source, tmp_path, control, report, delay=0)
    with pytest.raises(Cancelled):
        ex.run()
    assert len(d.taps) == 1


def test_unexpected_other_arrow_disappearance_is_rejected():
    source, b = scene()
    actual = np.full_like(source, (65, 55, 50))
    assert not ScreenVerifier(source,b).matches(actual, {2})


def test_tap_timeout_is_logged_as_uncertain_attempt(tmp_path):
    source, b = scene()
    d = FakeDevice(source)
    def fail(*args):
        raise RuntimeError("tap timeout")
    d.tap = fail
    events = []
    ex = Executor(d, b, solve(b), source, tmp_path, ExecutionControl(), lambda kind, _: events.append(kind))
    with pytest.raises(RuntimeError, match="timeout"):
        ex.run()
    assert "attempted" in events
    assert events[-1] == "halted"


def test_partial_recognition_cannot_execute(tmp_path):
    source, b = scene()
    b.arrows.pop(0)
    d = FakeDevice(source)
    with pytest.raises(ExecutionError, match="未解释"):
        Executor(d, b, solve(b), source, tmp_path, ExecutionControl())
    assert not d.taps


def test_background_click_point_rejected_before_tap(tmp_path):
    source,b = scene()
    b.arrows[0].click_point = (10,10)
    device = FakeDevice(source)
    with pytest.raises(ExecutionError,match="独立前景"):
        Executor(device,b,solve(b),source,tmp_path,ExecutionControl())
    assert not device.taps
