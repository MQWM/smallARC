from types import SimpleNamespace
from pathlib import Path

import pytest

from arrow_solver.continuous import ContinuousPlay
from arrow_solver.executor import Cancelled,ExecutionError
from arrow_solver.next_level import NextLevelResult
from test_executor import scene,FakeDevice
from test_next_level import ClockControl
from arrow_solver.vision import read_image


def setup_levels(monkeypatch,control,boards):
    calls = []
    image,_ = scene()
    levels = iter(boards)
    def advance(self):
        calls.append(1)
        if len(calls)>len(boards):
            raise Cancelled('test stopped after complete levels')
        return NextLevelResult(image.copy(),control.t)
    monkeypatch.setattr('arrow_solver.continuous.NextLevel.run',advance)
    monkeypatch.setattr('arrow_solver.continuous.recognize',lambda image:SimpleNamespace(board=next(levels)))
    return image,calls


def test_continues_multiple_levels_with_fresh_records_and_no_arrow_step_captures(tmp_path,monkeypatch):
    _,first = scene()
    _,second = scene()
    control = ClockControl()
    image,calls = setup_levels(monkeypatch,control,[first,second])
    device = FakeDevice(image)
    device.screenshot = lambda:pytest.fail('whole-plan execution must not capture between arrow taps')
    events = []
    runner = ContinuousPlay(device,control,tmp_path,lambda kind,data:events.append(kind),delay=0,now=lambda:control.t)
    with pytest.raises(Cancelled):
        runner.run()
    assert device.taps == [(220,60),(100,60)]*2
    assert len(calls) == 3
    assert events.count('auto_level_source') == 2
    assert events.count('auto_level_plan') == 2
    assert events.count('sent_complete') == 2
    assert len(list(tmp_path.glob('*/execution.jsonl'))) == 2


def test_real_screenshots_are_recognized_and_executed_across_levels(tmp_path,monkeypatch):
    fixtures = Path(__file__).parent/'fixtures'
    frames = iter([read_image(fixtures/'level2.jpg'),read_image(fixtures/'recorded/level199.png')])
    control = ClockControl()
    def advance(runner):
        try:
            image = next(frames)
        except StopIteration:
            raise Cancelled('test stopped')
        return NextLevelResult(image,control.t)
    monkeypatch.setattr('arrow_solver.continuous.NextLevel.run',advance)
    device = SimpleNamespace(serial='test-only',taps=[])
    device.tap = lambda *point:device.taps.append(point)
    device.screenshot = lambda:pytest.fail('unexpected mid-level screenshot')
    events = []
    runner = ContinuousPlay(device,control,tmp_path,lambda kind,data:events.append(kind),delay=0,now=lambda:control.t)
    with pytest.raises(Cancelled):
        runner.run()
    assert len(device.taps) == 59+127
    assert events.count('sent_complete') == 2
    assert len(list(tmp_path.glob('*/board.json'))) == 2


@pytest.mark.parametrize('failure',['recognition','plan','preflight','timer'])
def test_invalid_new_level_stops_without_any_arrow_click(tmp_path,monkeypatch,failure):
    _,board = scene()
    if failure == 'recognition':
        board.issues.append('头部不唯一')
    elif failure == 'plan':
        board.arrows[1].direction = 'LEFT'
        board.arrows[1].cells.reverse()
    elif failure == 'preflight':
        board.arrows[0].click_point = (10,10)
    else:
        board.arrows[0].kind = 'timer'
        board.arrows[0].visual_rect = board.arrows[0].head_rect = (50,40,65,40)
        board.timer_seconds = None
    control = ClockControl()
    image,calls = setup_levels(monkeypatch,control,[board])
    device = FakeDevice(image)
    runner = ContinuousPlay(device,control,tmp_path,lambda *_:None,delay=0,now=lambda:control.t)
    with pytest.raises((ExecutionError,ValueError)):
        runner.run()
    assert not device.taps
    assert len(calls) == 1
    assert len(list(tmp_path.glob('*/source.png'))) == 1


def test_stop_during_recognition_prevents_execution(tmp_path,monkeypatch):
    _,board = scene()
    control = ClockControl()
    image,calls = setup_levels(monkeypatch,control,[board])
    device = FakeDevice(image)
    def report(kind,data):
        if kind == 'auto_level_recognized':
            control.cancel.set()
    runner = ContinuousPlay(device,control,tmp_path,report,delay=0,now=lambda:control.t)
    with pytest.raises(Cancelled):
        runner.run()
    assert not device.taps
    assert len(calls) == 1


def test_new_level_timer_gets_its_own_first_tap_deadline(tmp_path,monkeypatch):
    boards = []
    for _ in range(2):
        _,board = scene()
        timer = board.arrows[0]
        timer.kind = 'timer'
        timer.visual_rect = timer.head_rect = (50,40,65,40)
        board.timer_seconds = 10
        boards.append(board)
    control = ClockControl()
    control.t = 1000.
    image,_ = setup_levels(monkeypatch,control,boards)
    device = FakeDevice(image)
    deadlines = []
    def report(kind,data):
        if kind == 'timer_started':
            deadlines.append(data['deadline'])
    runner = ContinuousPlay(device,control,tmp_path,report,delay=0,now=lambda:control.t)
    with pytest.raises(Cancelled):
        runner.run()
    assert deadlines == [1010.,1025.]
    assert device.taps == [(220,60)]*2
