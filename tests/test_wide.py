from pathlib import Path
import cv2
import numpy as np
import pytest

from arrow_solver.vision import read_image, recognize
from arrow_solver.models import Arrow, Board
from arrow_solver.solver import solve, occupancy, BoardError
from image_assertions import ScreenVerifier
from arrow_solver.renderer import export_run


@pytest.fixture(scope='module')
def wide():
    image = read_image(Path(__file__).parent/'fixtures/recorded/level202.png')
    return image,recognize(image).board


def test_real_wide_arrow_has_full_footprint_and_click(wide,tmp_path):
    image,board = wide
    assert not board.issues
    assert len(board.arrows) == 113
    arrows = [a for a in board.arrows if a.kind == 'wide']
    assert len(arrows) == 1
    arrow = arrows[0]
    assert arrow.direction == 'DOWN'
    assert {x for x,y in arrow.footprint} == {8,9,10}
    assert arrow.head == (9,34)
    plan = solve(board)
    assert len(plan.order) == 113
    verifier = ScreenVerifier(image,board)
    assert verifier.matches(image,set())
    frame=image.copy()
    roi=frame[verifier.roi]
    roi[verifier.masks[arrow.id]]=verifier.background
    assert verifier.matches(frame,{arrow.id})
    assert not verifier.matches(image,{arrow.id})
    path=export_run(tmp_path,image,board,plan)
    assert Board.load(path/'board.json').fingerprint()==board.fingerprint()


@pytest.mark.parametrize('scale',[.5,.75,1.25])
def test_wide_arrow_scales_with_board(wide,scale):
    image,board=wide
    other=recognize(cv2.resize(image,None,fx=scale,fy=scale)).board
    # Test the new wide shape independently: other low-resolution sprites may
    # still report issues, and must not be silently marked as recognized.
    signature=lambda b: {(tuple(a.cells),a.direction,a.kind,tuple(a.footprint)) for a in b.arrows if a.kind == "wide"}
    assert signature(other)==signature(board)


def test_off_center_blocker_blocks_wide_exit_and_ordinary_hits_wide_body():
    wide=Arrow(1,[(2,1),(2,2)],'DOWN',(0,0),kind='wide',
               footprint=[(x,y) for x in (1,2,3) for y in (1,2)])
    # #2 is not on the center ray but still obstructs the broad arrow.
    side=Arrow(2,[(3,4),(4,4)],'RIGHT',(0,0))
    # #3 hits the broad side although it does not meet its center path.
    normal=Arrow(3,[(0,2)],'RIGHT',(0,0))
    board=Board(200,200,(0,0,200,200),(0,0),(20,20),8,8,[wide,side,normal])
    plan=solve(board)
    assert plan.blockers[1]==[2]
    assert plan.blockers[3]==[1]
    assert plan.order==[2,1,3]


def test_burning_prefix_is_visual_only_and_does_not_free_cells(wide):
    image,board=wide
    timer=next(a for a in board.arrows if a.kind=='timer')
    verifier=ScreenVerifier(image,board)
    before=occupancy(board)
    frame=image.copy()
    points=np.rint([board.pixel(c) for c in timer.cells[:5]]).astype(np.int32)
    # Synthetic burnt tail and a small flame at the burning front.
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.polylines(mask,[points],False,255,round(board.line_width*3))
    frame[mask>0]=verifier.background
    cv2.circle(frame,tuple(points[-1]),round(board.line_width*.5),(20,140,255),-1)
    assert verifier.matches(frame,set(),burning=True)
    assert not verifier.matches(frame,set(),burning=False)
    assert occupancy(board)==before
    # Arbitrary disappearance in the middle, or loss of the missile head, is rejected.
    broken=image.copy()
    x,y=map(round,board.pixel(timer.cells[len(timer.cells)//2]))
    cv2.circle(broken,(x,y),round(board.line_width),tuple(map(int,verifier.background)),-1)
    # Remove a second later point so it cannot be a one-cell flame boundary.
    x,y=map(round,board.pixel(timer.cells[len(timer.cells)//2+1]))
    cv2.circle(broken,(x,y),round(board.line_width),tuple(map(int,verifier.background)),-1)
    assert not verifier.matches(broken,set(),burning=True)
    hx,hy,hw,hh=timer.head_rect
    frame[hy:hy+hh,hx:hx+hw]=verifier.background
    assert not verifier.matches(frame,set(),burning=True)
