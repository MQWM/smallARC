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
from arrow_solver.orbit import position,predicted_position,measure,capture_pair

FIXTURES = Path(__file__).parent/'fixtures/recorded'


def test_two_frame_measurement_uses_recorded_positions_and_roundtrips(tmp_path):
    first = detect(read_image(FIXTURES/'level332.png'))
    second = detect(read_image(FIXTURES/'level332-later.png'))
    measure(first,second,.968007)
    assert second.measured_clockwise is True
    assert second.measured_speed == pytest.approx(117.915,abs=1)
    assert predicted_position(second,.1,second.measured_speed,True)[1] == pytest.approx(second.center[1]-11.7915,abs=1)
    _,board = scene()
    board.orbit_obstacle = second
    board.save(tmp_path/'board.json')
    assert Board.load(tmp_path/'board.json').fingerprint() == board.fingerprint()


@pytest.mark.parametrize('start,end,clockwise',[(390,20,True),(20,390,False),(90,120,True)])
def test_measure_crosses_lap_and_corner(start,end,clockwise):
    rect = (0,0,100,100)
    first = OrbitObstacle(rect,position(rect,start),5,[])
    second = OrbitObstacle(rect,position(rect,end),5,[])
    measure(first,second,1)
    assert second.measured_speed == pytest.approx(30)
    assert second.measured_clockwise is clockwise
    assert phase(rect,predicted_position(second,1,30,clockwise)) == pytest.approx((end+(30 if clockwise else -30))%400)


def test_pair_only_captures_second_frame_when_bean_present():
    clock = ClockControl()
    images = [read_image(FIXTURES/'level332.png'),read_image(FIXTURES/'level332-later.png')]
    calls = []
    class Device:
        def screenshot(self):
            image = images[len(calls)]
            calls.append(clock.t)
            clock.t += .1
            return image
    sample = capture_pair(Device(),clock,now=lambda:clock.t)
    assert len(calls)==2 and not sample.warning
    assert sample.image is images[1]
    assert sample.captured_at == pytest.approx(calls[-1]+.05)
    assert sample.obstacle.sample_interval == pytest.approx(.85)
    assert sample.obstacle.measured_clockwise
    normal,_ = scene()
    images[:] = [normal]
    calls.clear()
    assert capture_pair(Device(),clock,now=lambda:clock.t).obstacle is None
    assert len(calls)==1


def test_sampling_failure_is_explicit_and_stop_prevents_second_capture():
    image = read_image(FIXTURES/'level332.png')
    clock = ClockControl()
    class Device:
        def screenshot(self):return image
    sample = capture_pair(Device(),clock,now=lambda:clock.t)
    assert sample.warning and sample.obstacle.measured_speed is None
    clock.cancel.set()
    with pytest.raises(Cancelled):
        capture_pair(Device(),clock,now=lambda:clock.t)


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
    assert wait_seconds(o,[5],0,10,3,True,0)>0
    assert wait_seconds(o,[5],0,10,3,False,0)==0
    assert wait_seconds(o,[5],3,10,3,True,0)==0


def test_exit_filter_predicts_near_future_not_only_current_distance():
    o=OrbitObstacle((0,0,100,100),(0,0),1,[])
    assert wait_seconds(o,[20],0,10,3,True,0)==pytest.approx(2.6)
    assert wait_seconds(o,[100],0,10,3,True,0)==0
    with pytest.raises(ValueError,match='过期'):
        wait_seconds(o,[20],10000,10,3)
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
    clock.t=1.2
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
                    at=(initial+118*(clock.t+.2+dt))%perimeter
                    gap=abs(at-target)
                    assert min(gap,perimeter-gap)>o.radius+board.line_width*.5
            seen.append(arrow.id)
            del remaining[arrow.id]
            clock.t+=.2
    ex=Executor(Device(),board,plan,image,tmp_path,clock,delay=.4,
                now=lambda:clock.t,captured_at=0)
    assert ex.run()==111
    assert seen==plan.order


@pytest.mark.parametrize('scale',[.75,1.,1.25])
def test_333_heart_on_rail_is_ignored_but_all_arrows_remain(scale):
    image=read_image(FIXTURES/'level333.png')
    image=cv2.resize(image,None,fx=scale,fy=scale)
    board=recognize(image).board
    assert board.orbit_obstacle and not board.issues
    assert len(board.arrows)==109
    assert len(solve(board).order)==109
    ScreenValidator(image,board)
    # The right rail's heart is excluded even where it protrudes into the crop.
    x,y,w,h=board.orbit_obstacle.track_rect
    assert any(bx<=x+w<=bx+bw and by<=950*scale<=by+bh
               for bx,by,bw,bh in board.orbit_obstacle.ignored_regions)


@pytest.mark.parametrize('color',[(180,180,180),(230,110,100),(170,70,210)])
def test_unknown_rail_items_are_ignored_without_color_specific_detection(color):
    from arrow_solver.vision import foreground_mask
    image=read_image(FIXTURES/'level333.png')
    original=detect(image)
    x,y,w,h=original.track_rect
    center=(round(x+w*.5),round(y))
    cv2.circle(image,center,20,color,-1)
    obstacle=detect(image)
    assert obstacle and obstacle.center==original.center
    mask=mask_objects(foreground_mask(image),(0,0,image.shape[1],image.shape[0]),obstacle)
    assert mask[center[1]+10,center[0]]==0
    # A valid arrow near the top remains foreground.
    assert mask[556,175]>0


def test_constant_speed_is_independent_of_track_perimeter():
    small=OrbitObstacle((0,0,100,100),(0,0),1,[])
    large=OrbitObstacle((0,0,200,200),(0,0),1,[])
    # Same physical distance takes the same time on two differently sized maps.
    assert wait_seconds(small,[20],0,speed=10,window=5,line_width=0)==pytest.approx(2.6)
    assert wait_seconds(large,[20],0,speed=10,window=5,line_width=0)==pytest.approx(2.6)
    # Twice the physical distance takes twice the travel time, plus the margin.
    assert wait_seconds(large,[40],0,speed=10,window=5,line_width=0)==pytest.approx(4.6)
