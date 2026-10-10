from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from image_assertions import ScreenVerifier
from arrow_solver.geometry import segment_distance, launch_segments
from arrow_solver.models import Arrow, Board, CounterBlock, Plan, Portal
from arrow_solver.solver import BoardError, solve, verify_plan
from arrow_solver.vision import read_image, recognize


def scene(arrows):
    return Board(200,200,(0,0,200,200),(0,0),(10,10),20,20,arrows,line_width=2.)


def oblique():
    return Arrow(2,[(6,6),(5,6),(5,5),(4,5),(4,4)],'UP_LEFT',(40,35),
                 pixel_path=[(60.,65.),(40.,35.)],launch_vectors=[(-2.,-3.)])


def test_non_grid_body_blocks_ordinary_ray_between_grid_points():
    moving = Arrow(1,[(1,5),(2,5)],'RIGHT',(20,50))
    board = scene([moving,oblique()])
    plan = solve(board)
    assert plan.blockers == {1:[2],2:[]}
    assert plan.order == [2,1]
    with pytest.raises(BoardError,match='仍被阻挡'):
        verify_plan(board,Plan([1,2],{},board.fingerprint()))


def test_both_candidate_launch_angles_must_clear():
    moving = Arrow(1,[(8,7),(7,7),(6,6),(5,5)],'UP_LEFT',(50,50),
                   pixel_path=[(80.,70.),(50.,50.)],launch_vectors=[(-1.,-1.),(-4.,-3.)])
    board = scene([moving,Arrow(2,[(1,1)],'LEFT',(10,10)),Arrow(3,[(1,2)],'LEFT',(10,20))])
    plan = solve(board)
    assert plan.blockers[1] == [2,3]
    assert plan.order[-1] == 1


@pytest.mark.parametrize('gap,blocked',[(1.9,True),(2.1,False)])
def test_channel_and_body_width_both_count(gap,blocked):
    moving = Arrow(1,[(1,5),(2,5)],'RIGHT',(20,50))
    obstacle = Arrow(2,[(6,5),(7,5)],'RIGHT',(70,50),
                     pixel_path=[(60.,50.+gap),(70.,50.+gap)],launch_vectors=[(1.,0.)])
    assert (2 in solve(scene([moving,obstacle])).blockers[1]) == blocked


def test_continuous_ray_stops_at_portal_before_obstacle():
    moving = oblique()
    obstacle = Arrow(3,[(2,0),(2,1)],'DOWN',(20,10))
    board = scene([moving,obstacle])
    assert 3 in solve(board).blockers[2]
    board.portals = [Portal((3,2),(30.,20.),(26,16,8,8))]
    assert solve(board).blockers[2] == []


def test_geometry_counter_requires_successful_click_before_release():
    moving = oblique()
    board = scene([moving,Arrow(3,[(10,10)],'RIGHT',(100,100))])
    board.counter_blocks = [CounterBlock((3,2),1,(27,17,6,6))]
    assert solve(board).order == [3,2]
    with pytest.raises(BoardError,match='尚未解锁'):
        verify_plan(board,Plan([2,3],{},board.fingerprint()))


def test_missing_geometry_and_wrong_launch_direction_are_rejected():
    arrow = oblique()
    arrow.pixel_path = []
    with pytest.raises(BoardError,match='真实折线路径缺失'):
        solve(scene([arrow]))
    arrow = oblique()
    arrow.launch_vectors = [(1.,0.)]
    with pytest.raises(BoardError,match='发射向量与头部方向不一致'):
        solve(scene([arrow]))


def test_segment_distance_handles_crossings_and_collinear_points():
    assert segment_distance((0,0),(10,8),(5,-5),(5,20)) == 0
    assert segment_distance((0,0),(10,0),(5,0),(15,0)) == 0
    assert segment_distance((0,0),(0,0),(3,4),(3,4)) == 5


@pytest.fixture(scope='module')
def level246():
    image = read_image(Path(__file__).parent/'fixtures/recorded/level246.png')
    return image,recognize(image).board


def test_level246_real_polyline_plan_and_visual_preflight(level246,tmp_path):
    image,board = level246
    assert not board.issues
    assert len(board.arrows) == 79
    arrow = next(a for a in board.arrows if a.id == 32)
    assert len(arrow.launch_vectors) == 2
    delta = np.array(arrow.pixel_path[-1])-arrow.pixel_path[-2]
    assert 37 < np.degrees(np.arctan2(abs(delta[1]),abs(delta[0]))) < 40
    plan = solve(board)
    assert len(plan.order) == 79
    assert plan.blockers[32] == [2,4,12,17,23,29]
    assert plan.order.index(32) == 10
    verifier = ScreenVerifier(image,board)
    assert verifier.matches(image,set())
    # Independent raster check against original foreground instances, rather
    # than reusing continuous segment-distance calculations as the test oracle.
    beam = np.zeros(image.shape[:2],np.uint8)
    for start,end,radius in launch_segments(board,arrow):
        cv2.line(beam,tuple(map(round,start)),tuple(map(round,end)),255,round(radius*2))
    raster_hits = sorted(aid for aid,mask in verifier.masks.items() if aid != 32 and
                         np.count_nonzero(mask & (beam[verifier.roi]>0)) > 10)
    assert raster_hits == plan.blockers[32]
    frame = image.copy()
    frame[verifier.roi] = verifier.expected({32})
    assert verifier.matches(frame,{32})
    assert not verifier.matches(image,{32})
    board.save(tmp_path/'board.json')
    loaded = Board.load(tmp_path/'board.json')
    assert loaded.fingerprint() == board.fingerprint()
    assert solve(loaded).order == plan.order
    # Keeping a staircase while dropping continuous metadata must remain illegal.
    broken = deepcopy(board)
    broken.arrows[31].launch_vectors = []
    with pytest.raises(BoardError,match='头部方向与末段路径不一致'):
        solve(broken)


@pytest.mark.parametrize('scale',[1.25])
def test_level246_scaled_continuous_geometry(level246,scale):
    image,_ = level246
    resized = cv2.resize(image,None,fx=scale,fy=scale)
    board = recognize(resized).board
    assert not board.issues
    assert len(board.arrows) == 79
    assert len(solve(board).order) == 79
    skew = next(a for a in board.arrows if len(a.launch_vectors) == 2)
    assert abs(np.linalg.norm(np.array(skew.click_point)-np.array((364,1032))*scale)) < 4
    assert ScreenVerifier(resized,board).matches(resized,set())


def test_level246_rejects_merged_heads_at_low_resolution(level246):
    image,_ = level246
    # At 75% two physically separate arrow tips merge through antialiasing.
    # Preserve the ambiguity rejection instead of selecting a convenient head.
    board = recognize(cv2.resize(image,None,fx=.75,fy=.75)).board
    assert any('多个可能的头部' in issue for issue in board.issues)
