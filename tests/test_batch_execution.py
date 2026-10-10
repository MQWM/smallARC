import json

import pytest

from arrow_solver.executor import Executor, ExecutionControl, ExecutionError, Cancelled
from arrow_solver.solver import solve
from test_executor import scene, FakeDevice


class CountingDevice(FakeDevice):
    def __init__(self, image, fail=False):
        super().__init__(image, fail)
        self.captures = 0

    def screenshot(self):
        self.captures += 1
        return super().screenshot()


def test_batch_does_not_claim_success_even_if_clicks_have_no_effect(tmp_path):
    source, board = scene()
    device = CountingDevice(source, fail=True)
    ex = Executor(device, board, solve(board), source, tmp_path, ExecutionControl(), delay=0)
    assert ex.run() == 2
    assert device.captures == 0
    log = [json.loads(line) for line in (tmp_path/'execution.jsonl').read_text().splitlines()]
    assert not any(event['kind'] in ('verified','complete') for event in log)


def test_batch_stop_after_dispatch_resumes_without_repeating_tap(tmp_path):
    source, board = scene()
    device = CountingDevice(source)
    control = ExecutionControl()
    cursor = []
    def report(kind, data):
        if kind == 'dispatched':
            cursor.append(data['step'])
            control.cancel.set()
    ex = Executor(device, board, solve(board), source, tmp_path, control, report, delay=0)
    with pytest.raises(Cancelled):
        ex.run()
    ex.control = ExecutionControl()
    ex.report = lambda *_: None
    assert ex.run(start=cursor[-1]) == 2
    assert device.taps == [(220,60),(100,60)]
    assert device.captures == 0


class ClockControl:
    def __init__(self):
        self.t = 0.
        self.waits = []
    def now(self):
        return self.t
    def check(self):
        pass
    def wait(self, seconds):
        self.waits.append(seconds)
        self.t += seconds


def timer_scene():
    source, board = scene()
    # Left arrow is a timed instance blocked by the right arrow.
    board.arrows[0].kind = 'timer'
    board.arrows[0].visual_rect = (50,40,65,40)
    board.arrows[0].head_rect = (50,40,65,40)
    return source, board


def test_timer_waits_without_tapping_and_keeps_blocker_dependency(tmp_path):
    source, board = timer_scene()
    clock = ClockControl()
    device = CountingDevice(source)
    events = []
    plan = solve(board)
    assert plan.order == [2,1]
    ex = Executor(device, board, plan, source, tmp_path, clock,
                  lambda kind, _: events.append(kind), delay=.1,
                  timer_deadline=10, flight_wait=3, now=clock.now)
    assert ex.run() == 2
    assert device.taps == [(220,60)]
    assert clock.t == 13
    assert device.captures == 0
    assert 'waiting_timer' in events
    assert 'assumed_departed' in events


@pytest.mark.parametrize('deadline', [None, -5, .2])
def test_timer_missing_or_expired_deadline_stops_before_click(tmp_path, deadline):
    source, board = timer_scene()
    clock = ClockControl()
    device = CountingDevice(source)
    ex = Executor(device, board, solve(board), source, tmp_path, clock,
                  timer_deadline=deadline, now=clock.now)
    with pytest.raises(ExecutionError):
        ex.run()
    assert device.taps == []
    assert device.captures == 0


def test_fresh_countdown_begins_at_first_tap_and_survives_resume(tmp_path):
    source, board = timer_scene()
    clock = ClockControl()
    clock.t = 1000  # arbitrary time spent reading/recognizing before the first click
    device = CountingDevice(source)
    events = []
    def report(kind,data):
        events.append((kind,data))
        if kind == 'dispatched':
            raise Cancelled('stopped after dispatch')
    ex = Executor(device, board, solve(board), source, tmp_path, clock,
                  report, delay=.1,
                  timer_duration=10, flight_wait=3, now=clock.now)
    with pytest.raises(Cancelled):
        ex.run()
    assert ex.timer_deadline == 1010
    deadline = next(data['deadline'] for kind,data in events if kind == 'timer_started')
    clock.t += 2
    resumed = Executor(device, board, solve(board), source, tmp_path, clock,
                       timer_deadline=deadline, timer_duration=10,
                       flight_wait=3, now=clock.now)
    assert resumed.run(start=1) == 2
    assert clock.t == 1013
    assert device.taps == [(220,60)]


def test_unstarted_timer_without_any_tap_does_not_assume_departure(tmp_path):
    source, board = timer_scene()
    clock = ClockControl()
    device = CountingDevice(source)
    ex = Executor(device, board, solve(board), source, tmp_path, clock,
                  timer_duration=10, now=clock.now)
    with pytest.raises(ExecutionError, match="尚未启动"):
        ex.run(start=1)
    assert not device.taps


def test_complete_plan_sends_without_captures(tmp_path):
    source,board = scene()
    device = CountingDevice(source)
    events = []
    ex = Executor(device,board,solve(board),source,tmp_path,ExecutionControl(),
                  lambda kind,data: events.append(kind),delay=0)
    assert ex.run() == 2
    assert device.taps == [(220,60),(100,60)]
    assert device.captures == 0
    assert events.count('dispatched') == 2
    assert 'verified' not in events
    assert events[-1] == 'sent_complete'
