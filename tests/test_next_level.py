from pathlib import Path

import cv2
import numpy as np
import pytest

from arrow_solver.executor import ExecutionControl,ExecutionError,Cancelled
from arrow_solver.next_level import NextLevel,next_button,new_board,_header
from arrow_solver.vision import read_image


ROOT = Path(__file__).parent/'fixtures/recorded'


@pytest.fixture(scope='module')
def victory():
    return read_image(ROOT/'level262-victory.png')


@pytest.mark.parametrize('scale',[.5,.75,1.,1.25])
def test_victory_button_tracks_screenshot_size(victory,scale):
    frame = cv2.resize(victory,None,fx=scale,fy=scale)
    point = next_button(frame)
    assert point is not None
    assert np.linalg.norm(np.array(point)-np.array((540,1773))*scale)<2


def test_button_requires_victory_heading(victory):
    frame = victory.copy()
    frame[450:750] = 0
    assert next_button(frame) is None
    assert next_button(read_image(ROOT/'level246.png')) is None
    frame = victory.copy()
    frame[1735:1820,435:650] = (160,186,86)
    assert next_button(frame) is None  # Same green button, missing next-level text.


def test_new_board_requires_changed_header_and_real_arrows(victory):
    image = read_image(ROOT/'level246.png')
    assert new_board(image,_header(victory))
    assert not new_board(image,_header(image))
    dimmed = (image*.6).astype(np.uint8)
    assert not new_board(image,_header(dimmed))
    assert not new_board(np.zeros_like(image),_header(victory))
    # A loading screen with the new title but no actual board is not entry.
    loading = image.copy()
    loading[round(image.shape[0]*.15):] = (65,55,50)
    assert not new_board(loading,_header(victory))


class ClockControl(ExecutionControl):
    def __init__(self):
        super().__init__()
        self.t = 0.
        self.waits = []

    def wait(self,seconds):
        self.check()
        self.waits.append(seconds)
        self.t += seconds
        self.check()


class Device:
    def __init__(self,frames):
        self.frames = iter(frames)
        self.taps = []
        self.captures = 0

    def screenshot(self):
        self.captures += 1
        return next(self.frames)

    def tap(self,*point):
        self.taps.append(point)


def test_checks_every_five_seconds_and_never_repeats_next_tap(victory):
    board = read_image(ROOT/'level246.png')
    device = Device([victory,victory,board])
    control = ClockControl()
    result = NextLevel(device,control,now=lambda:control.t).run()
    assert control.waits == [5.,5.]
    assert device.taps == [(540,1773)]
    assert device.captures == 3
    assert result.image is board
    assert result.captured_at == 10.


def test_waits_for_actual_victory_before_clicking(victory):
    board = read_image(ROOT/'level246.png')
    device = Device([board,victory,board])
    control = ClockControl()
    NextLevel(device,control,now=lambda:control.t).run()
    assert device.taps == [(540,1773)]
    assert control.waits == [5.,5.]


def test_stop_after_next_tap_prevents_more_captures(victory):
    device = Device([victory])
    control = ClockControl()
    def report(kind,data):
        if kind == 'next_level_clicked':
            control.cancel.set()
    with pytest.raises(Cancelled):
        NextLevel(device,control,report,now=lambda:control.t).run()
    assert device.captures == 1
    assert len(device.taps) == 1


def test_timeout_does_not_click_loading_screen():
    device = Device([np.zeros((2400,1080,3),np.uint8)]*3)
    control = ClockControl()
    with pytest.raises(ExecutionError,match='15 秒'):
        NextLevel(device,control,timeout=15,now=lambda:control.t).run()
    assert not device.taps
    assert device.captures == 3
