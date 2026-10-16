from pathlib import Path
from copy import deepcopy
import cv2
import pytest
from arrow_solver.models import Arrow,Board,CounterBlock,Plan
from arrow_solver.solver import solve,verify_plan,BoardError,occupancy
from arrow_solver.vision import recognize,read_image
from image_assertions import ScreenVerifier
from arrow_solver.executor import Executor, ExecutionControl, ExecutionError
from path_helpers import expand_vertices
from test_executor import scene as fake_scene,FakeDevice


def make(arrows,blocks=[]):
    return Board(200,200,(0,0,200,200),(10,10),(20,20),9,9,arrows,counter_blocks=blocks)


def arrow(aid,cells,d='RIGHT'):
    return Arrow(aid,cells,d,(0,0))


def test_obstacle_waits_for_successful_count_without_becoming_a_click():
    b=make([arrow(1,[(0,1),(1,1)]),arrow(2,[(6,4),(7,4)]),arrow(3,[(6,6),(7,6)])],
           [CounterBlock((3,1),2,(60,20,20,20))])
    p=solve(b)
    assert p.order==[2,3,1]
    with pytest.raises(BoardError,match='尚未解锁'):
        verify_plan(b,Plan([1,2,3],p.blockers,b.fingerprint()))
    b.counter_blocks[0].remaining=3
    with pytest.raises(BoardError,match='没有可消除'):
        solve(b)


def test_diagonal_ray_and_midsegment_collision():
    b=make([arrow(1,[(0,0),(1,1)],'DOWN_RIGHT'),arrow(2,[(3,2),(2,3)],'DOWN_LEFT')])
    # Exit ray crosses the diagonal segment at (2.5,2.5), not a lattice endpoint.
    assert occupancy(b)[(2.5,2.5)]==2
    assert solve(b).blockers[1]==[2]
    assert solve(b).order==[2,1]
    assert expand_vertices([[0,0],[2,2],[2,4]])==[(0,0),(1,1),(2,2),(2,3),(2,4)]


@pytest.fixture(scope='module')
def real():
    image=read_image(Path(__file__).parent/'fixtures/recorded/level205.png')
    return image,recognize(image).board


def test_real_new_shapes_and_counter_thresholds(real,tmp_path):
    image,b=real
    assert not b.issues
    assert len(b.arrows)==153
    assert sorted(block.remaining for block in b.counter_blocks)==[40,80]
    assert {a.direction for a in b.arrows if '_' in a.direction}=={'DOWN_RIGHT','DOWN_LEFT'}
    assert sum(a.kind=='electric' for a in b.arrows)==1
    p=solve(b)
    assert len(p.order)==153
    assert ScreenVerifier(image,b).matches(image,set())
    b.save(tmp_path/'board.json')
    assert Board.load(tmp_path/'board.json').fingerprint()==b.fingerprint()


def test_block_digits_change_and_only_disappear_at_zero():
    from arrow_solver.special import read_white_number
    import numpy as np
    image,b=fake_scene()
    block=CounterBlock((0,3),1,(5,120,32,35))
    b.counter_blocks=[block]
    # Draw an independent compact numbered obstacle away from both arrows.
    cv2.rectangle(image,(5,120),(36,154),(20,175,250),-1)
    cv2.putText(image,'1',(13,147),cv2.FONT_HERSHEY_DUPLEX,.8,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(image,'1',(13,147),cv2.FONT_HERSHEY_DUPLEX,.8,(255,255,255),1,cv2.LINE_AA)
    assert read_white_number(image[120:155,5:37])==1
    v=ScreenVerifier(image,b)
    assert v.matches(image,set())
    after=image.copy()
    after[v.roi][v.masks[2]]=v.background
    assert not v.matches(after,{2})  # number should have reached zero and vanished
    after[118:157,3:39]=v.background
    assert v.matches(after,{2})
    assert not v.matches(after,set())


def test_electric_batch_continues_without_captures(tmp_path):
    image,b=fake_scene()
    b.arrows[1].kind='electric'
    device=FakeDevice(image)
    calls=[]
    device.screenshot=lambda: calls.append(1)
    ex=Executor(device,b,solve(b),image,tmp_path,ExecutionControl(),delay=0)
    assert ex.run()==2
    assert len(device.taps)==2
    assert not calls


def test_level220_full_paths_numbered_tiles_and_execution_preflight():
    image=read_image(Path(__file__).parent/'fixtures/recorded/level220.png')
    board=recognize(image).board
    assert not board.issues
    assert len(board.arrows)==200
    assert sorted(block.remaining for block in board.counter_blocks)==[47,167]
    plan=solve(board)
    assert len(plan.order)==200
    from arrow_solver.solver import block_threshold
    for arrow in board.arrows:
        assert plan.order.index(arrow.id)>=block_threshold(board,arrow)
    # The 167 tile's padded comparison area touches a blue arrow tip; it must
    # not claim that arrow's connected label or bypass its verification.
    verifier=ScreenVerifier(image,board)
    assert 51 in verifier.masks
    assert verifier.matches(image,set())
    assert not verifier.matches(image,{51})


@pytest.mark.parametrize('scale',[.75,1.,1.25])
@pytest.mark.parametrize('rect,expected',[( (22,972,31,28),167),((1029,1840,27,28),47)])
def test_game_counter_font_keeps_digits_at_different_sizes(scale,rect,expected):
    from arrow_solver.special import read_white_number
    image=read_image(Path(__file__).parent/'fixtures/recorded/level220.png')
    x,y,w,h=rect
    patch=cv2.resize(image[y:y+h,x:x+w],None,fx=scale,fy=scale)
    assert read_white_number(patch)==expected


@pytest.mark.parametrize('scale',[.75,1.,1.25])
@pytest.mark.parametrize('rect,expected',[((205,809,29,29),91),((514,1688,29,29),37)])
def test_level423_small_counter_digits_are_not_truncated(scale,rect,expected):
    from arrow_solver.special import read_white_number
    image = read_image(Path(__file__).parent/'fixtures/recorded/level423.png')
    x,y,w,h = rect
    patch = cv2.resize(image[y:y+h,x:x+w],None,fx=scale,fy=scale)
    assert read_white_number(patch)==expected


def test_misread_leading_zero_is_unknown_not_a_smaller_counter(monkeypatch):
    from arrow_solver import special
    image = read_image(Path(__file__).parent/'fixtures/recorded/level423.png')
    # Without a suitable 9 template this glyph used to become "01" -> 1.
    templates = [(d,t) for d,t in special.counter_digit_templates() if d!=9]
    monkeypatch.setattr(special,'counter_digit_templates',lambda:templates)
    assert special.read_white_number(image[809:838,205:234]) is None
    board = recognize(image).board
    assert any(b.remaining is None for b in board.counter_blocks)
    assert any('数字不确定' in issue for issue in board.issues)
    with pytest.raises(ValueError):
        solve(board)


@pytest.mark.parametrize('scale',[.75,1.,1.25])
def test_level423_counter_gates_delay_all_affected_actions(scale,tmp_path):
    from arrow_solver.executor import ScreenValidator,Executor
    from arrow_solver.solver import block_threshold,verify_plan
    from test_next_level import ClockControl
    image = cv2.resize(read_image(Path(__file__).parent/'fixtures/recorded/level423.png'),None,fx=scale,fy=scale)
    board = recognize(image).board
    assert not board.issues and len(board.arrows)==124
    assert sorted(b.remaining for b in board.counter_blocks)==[37,91]
    plan = solve(board)
    verify_plan(board,plan)
    for arrow in board.arrows:
        assert plan.order.index(arrow.id)>=block_threshold(board,arrow)
    affected = [a for a in board.arrows if block_threshold(board,a)==91]
    assert len(affected)==3
    if scale==1:
        assert plan.order.index(9)+1==92
        assert plan.order.index(27)+1==121
    ScreenValidator(image,board)
    clock = ClockControl()
    taps = []
    class Device:
        serial = 'test-only'
        def tap(self,*point):taps.append(point)
        def screenshot(self):pytest.fail('counter execution must not capture mid-level')
    assert Executor(Device(),board,plan,image,tmp_path,clock,delay=0,now=lambda:clock.t).run()==124
    assert len(taps)==124
