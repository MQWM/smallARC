from pathlib import Path
import cv2
import numpy as np
import pytest
from arrow_solver.models import Board,OrbitObstacle,Portal
from arrow_solver.orbit import detect,phase,wait_seconds,exit_phases,mask_objects
from arrow_solver.vision import read_image,recognize,suggest_crop
from arrow_solver.solver import solve
from arrow_solver.executor import Executor,ScreenValidator,ExecutionError,Cancelled
from test_executor import scene,FakeDevice
from test_next_level import ClockControl

FIXTURES = Path(__file__).parent/'fixtures/recorded'


@pytest.mark.parametrize('scale',[.75,1.,1.25])
def test_332_moving_objects_do_not_change_crop_or_arrow_paths(scale,tmp_path):
    boards=[]
    for filename in ('level332.png','level332-later.png'):
        image=read_image(FIXTURES/filename)
        image=cv2.resize(image,None,fx=scale,fy=scale)
        board=recognize(image).board
        assert board.orbit_obstacle and not board.issues
        assert len(board.arrows)==111
        assert len(solve(board).order)==111
        ScreenValidator(image,board)
        boards.append(board)
    assert boards[0].crop_rect==boards[1].crop_rect
    assert [a.cells for a in boards[0].arrows]==[a.cells for a in boards[1].arrows]
    boards[0].save(tmp_path/'board.json')
    assert Board.load(tmp_path/'board.json').fingerprint()==boards[0].fingerprint()
    expected_up=114*scale
    assert abs(boards[0].orbit_obstacle.center[1]-boards[1].orbit_obstacle.center[1]-expected_up)<3


def test_clockwise_phase_and_wrap_direction():
    rect=(0,0,100,100)
    assert [phase(rect,p) for p in [(0,0),(100,0),(100,100),(0,100),(0,10)]]==[0,100,200,300,390]
    o=OrbitObstacle(rect,(0,10),1,[])
    assert wait_seconds(o,[5],0,40,3,True,0)>0
    assert wait_seconds(o,[5],0,40,3,False,0)==0
    assert wait_seconds(o,[5],3,40,3,True,0)==0


def test_exit_filter_predicts_near_future_not_only_current_distance():
    o=OrbitObstacle((0,0,100,100),(0,0),1,[])
    assert wait_seconds(o,[20],0,40,3,True,0)==pytest.approx(2.6)
    assert wait_seconds(o,[100],0,40,3,True,0)==0
    with pytest.raises(ValueError,match='过期'):
        wait_seconds(o,[20],10000,40,3)
    with pytest.raises(ValueError,match='无效'):
        wait_seconds(o,[20],0,0,3)


def test_mask_outside_manual_crop_does_not_erase_arrows():
    mask=np.ones((30,30),np.uint8)
    o=OrbitObstacle((0,0,100,100),(0,0),5,[(0,0,10,10)])
    assert mask_objects(mask,(50,50,30,30),o).all()


def orbit_scene():
    image,board=scene()
    board.orbit_obstacle=OrbitObstacle((0,0,235,160),(235,60),5,[])
    return image,board


def test_executor_waits_before_tap_and_keeps_single_capture_contract(tmp_path):
    image,board=orbit_scene()
    clock=ClockControl()
    taps=[]
    events=[]
    class Device:
        serial='fake'
        def tap(self,*point):taps.append((point,clock.t))
        def screenshot(self):pytest.fail('unexpected capture')
    ex=Executor(Device(),board,solve(board),image,tmp_path,clock,
                lambda kind,data:events.append(kind),delay=0,now=lambda:clock.t,captured_at=0)
    assert ex.run()==2
    assert 'waiting_orbit' in events and taps[0][1]>.5
    assert [p for p,_ in taps]==[(220,60),(100,60)]
    # A portal absorbs the arrow before it reaches the rail.
    board.portals=[Portal((3,1),(130,60),(125,55,10,10))]
    assert exit_phases(board,board.arrows[0],{})==[]


def test_stop_during_avoidance_sends_no_click_and_resume_uses_original_capture_time(tmp_path):
    image,board=orbit_scene()
    clock=ClockControl()
    device=FakeDevice(image)
    def report(kind,data):
        if kind=='waiting_orbit':clock.cancel.set()
    ex=Executor(device,board,solve(board),image,tmp_path,clock,report,
                now=lambda:clock.t,captured_at=0)
    with pytest.raises(Cancelled):ex.run()
    assert not device.taps
    clock.cancel.clear()
    clock.t=5
    events=[]
    resumed=Executor(device,board,solve(board),image,tmp_path,clock,
                     lambda kind,data:events.append(kind),now=lambda:clock.t,captured_at=0)
    assert resumed.run()==2
    assert 'waiting_orbit' not in events


def test_missing_capture_timestamp_prevents_motion_execution(tmp_path):
    image,board=orbit_scene()
    with pytest.raises(ExecutionError,match='抓图时间'):
        Executor(FakeDevice(image),board,solve(board),image,tmp_path,ClockControl())


def test_332_all_111_taps_keep_nominal_exit_window_clear_without_captures(tmp_path):
    image=read_image(FIXTURES/'level332.png')
    board=recognize(image).board
    plan=solve(board)
    clock=ClockControl()
    clock.t=10  # Recognition and human review happened after the screenshot.
    o=board.orbit_obstacle
    perimeter=2*sum(o.track_rect[2:])
    initial=phase(o.track_rect,o.center)
    remaining={a.id:a for a in board.arrows}
    seen=[]
    class Device:
        serial='fake'
        def screenshot(self):pytest.fail('unexpected capture')
        def tap(self,*point):
            arrow=next(a for a in remaining.values() if a.click_point==point)
            for target in exit_phases(board,arrow,{}):
                # Check the complete nominal future window, including command latency.
                for dt in np.linspace(0,3,61):
                    at=(initial+perimeter/37*(clock.t+.2+dt))%perimeter
                    gap=abs(at-target)
                    assert min(gap,perimeter-gap)>o.radius+board.line_width*.5
            seen.append(arrow.id)
            del remaining[arrow.id]
            clock.t+=.2
    ex=Executor(Device(),board,plan,image,tmp_path,clock,delay=.4,
                now=lambda:clock.t,captured_at=0)
    assert ex.run()==111
    assert seen==plan.order
