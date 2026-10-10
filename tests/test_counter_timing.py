from pathlib import Path

import cv2
import pytest

from arrow_solver.executor import Executor,ExecutionError,Cancelled,ScreenValidator
from arrow_solver.models import CounterBlock,Arrow,Board
from arrow_solver.solver import solve,verify_plan,block_threshold
from arrow_solver.vision import read_image,recognize
from test_executor import scene
from test_next_level import ClockControl


def counter_scene():
    image,board = scene()
    board.counter_blocks = [CounterBlock((3,1),1,(130,50,20,20))]
    cv2.rectangle(image,(130,50),(149,69),(20,175,250),-1)
    return image,board


class TimedDevice:
    serial = 'test-only'

    def __init__(self,board,clock,flight=1.5):
        self.board,self.clock,self.flight = board,clock,flight
        self.completions,self.failed,self.taps = [],[],[]
        self.by_point = {a.click_point:a for a in board.arrows}

    def screenshot(self):
        pytest.fail('counter waiting must not add screenshots')

    def tap(self,*point):
        arrow = self.by_point[point]
        self.taps.append((arrow.id,self.clock.t))
        completed = sum(t<=self.clock.t+1e-9 for t in self.completions)
        if block_threshold(self.board,arrow)>completed:
            self.failed.append(arrow.id)
        else:
            self.completions.append(self.clock.t+self.flight)


def test_numbered_lane_is_postponed_while_other_free_arrows_exist():
    arrows = [Arrow(1,[(0,1),(1,1)],'RIGHT',(0,0)),
              Arrow(2,[(5,3),(6,3)],'RIGHT',(0,0)),
              Arrow(3,[(5,5),(6,5)],'RIGHT',(0,0))]
    board = Board(200,200,(0,0,200,200),(10,10),(20,20),9,9,arrows,
                  counter_blocks=[CounterBlock((3,1),1,(60,20,20,20))])
    assert solve(board).order == [2,3,1]


def test_delay_is_measured_from_threshold_dispatch_not_arrow_step_count(tmp_path):
    image,board = counter_scene()
    clock = ClockControl()
    device = TimedDevice(board,clock)
    events = []
    ex = Executor(device,board,solve(board),image,tmp_path,clock,
                  lambda kind,data:events.append((kind,data)),delay=.4,now=lambda:clock.t)
    assert ex.run() == 2
    assert device.failed == []
    assert device.taps == [(2,0.),(1,2.)]
    waiting = [data for kind,data in events if kind=='waiting_counter']
    assert len(waiting)==1 and waiting[0]['seconds']==pytest.approx(1.6)


def test_stopping_counter_wait_resumes_without_repeating_sent_arrow(tmp_path):
    image,board = counter_scene()
    clock = ClockControl()
    device = TimedDevice(board,clock)
    def report(kind,data):
        if kind == 'waiting_counter':
            clock.cancel.set()
    ex = Executor(device,board,solve(board),image,tmp_path,clock,report,delay=.4,now=lambda:clock.t)
    with pytest.raises(Cancelled):
        ex.run()
    assert device.taps == [(2,0.)]
    clock.cancel.clear()
    resumed = Executor(device,board,solve(board),image,tmp_path,clock,delay=.4,
                       not_before=ex.not_before,now=lambda:clock.t)
    assert resumed.run(start=1) == 2
    assert device.taps == [(2,0.),(1,2.)]
    assert not device.failed


def test_counter_wait_is_included_in_missile_budget(tmp_path):
    image,board = counter_scene()
    timer = board.arrows[0]
    timer.kind = 'timer'
    timer.visual_rect = timer.head_rect = (50,40,65,40)
    clock = ClockControl()
    device = TimedDevice(board,clock)
    ex = Executor(device,board,solve(board),image,tmp_path,clock,delay=.4,
                  timer_duration=2,now=lambda:clock.t)
    with pytest.raises(ExecutionError,match='清障时间不足'):
        ex.run()
    assert not device.taps
    # All required taps may already have been sent, while the final counter
    # still needs its margin. Resuming must check that wait against the timer too.
    clock.t = 9.5
    resumed = Executor(device,board,solve(board),image,tmp_path/'resume',clock,delay=.4,
                       timer_deadline=10,not_before=9.9,now=lambda:clock.t)
    with pytest.raises(ExecutionError,match='清障时间不足'):
        resumed.run(start=1)
    assert not device.taps


def test_level266_planning_and_delayed_counter_unlocks(tmp_path):
    image = read_image(Path(__file__).parent/'fixtures/recorded/level266.png')
    board = recognize(image).board
    assert not board.issues
    assert len(board.arrows)==97
    assert sorted(b.remaining for b in board.counter_blocks)==[19,39,52,72]
    plan = solve(board)
    verify_plan(board,plan)
    ScreenValidator(image,board)
    assert plan.order.index(43)+1==22
    assert plan.order.index(69)+1==53  # No extra free move exists: must wait here.
    events = []
    clock = ClockControl()
    device = TimedDevice(board,clock)
    ex = Executor(device,board,plan,image,tmp_path,clock,
                  lambda kind,data:events.append((kind,data)),delay=.4,now=lambda:clock.t)
    assert ex.run()==97
    assert len(device.taps)==97
    assert not device.failed
    assert {data['threshold'] for kind,data in events if kind=='waiting_counter'}=={19,39,52,72}
    # Independent device model reproduces premature unlocks without the margin.
    clock = ClockControl()
    unsafe = TimedDevice(board,clock)
    Executor(unsafe,board,plan,image,tmp_path/'without-margin',clock,delay=.4,
             counter_settle=0,now=lambda:clock.t).run()
    assert unsafe.failed
