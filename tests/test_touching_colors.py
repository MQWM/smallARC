from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import pytest

from arrow_solver.vision import read_image,recognize,foreground_mask
from arrow_solver.special import arrow_components
from arrow_solver.solver import solve,verify_plan,BoardError
from arrow_solver.models import Board
from arrow_solver.executor import ScreenValidator,ExecutionError

FIXTURE = Path(__file__).parent/'fixtures/recorded/level419.png'


@pytest.mark.parametrize('scale',[.75,1.,1.25])
def test_419_touching_heads_keep_both_complete_paths_and_collision_dependencies(scale,tmp_path):
    image = cv2.resize(read_image(FIXTURE),None,fx=scale,fy=scale)
    board = recognize(image).board
    assert len(board.arrows)==74 and not board.issues
    if scale==.75:
        # Both skins are recovered, but this resolution still quantizes a bend
        # of the oblique arrow into a repeated display cell. Keep the rejection.
        with pytest.raises(BoardError,match='路径重复经过同一单元'):
            solve(board)
        return
    plan = solve(board)
    verify_plan(board,plan)
    validator = ScreenValidator(image,board)
    pink = next(a for a in board.arrows if a.direction=='DOWN_LEFT' and a.click_point[1]>1700*scale)
    purple = next(a for a in board.arrows if a.direction=='LEFT' and a.click_point[1]>1700*scale and a.click_point[0]<200*scale)
    assert pink.id in plan.blockers[purple.id]
    assert plan.order.index(pink.id)<plan.order.index(purple.id)
    assert not np.any(validator.masks[pink.id]&validator.masks[purple.id])
    board.save(tmp_path/'board.json')
    assert Board.load(tmp_path/'board.json').fingerprint()==board.fingerprint()
    missing = deepcopy(board)
    missing.arrows = [a for a in missing.arrows if a.id!=purple.id]
    with pytest.raises(ExecutionError,match='未解释'):
        ScreenValidator(image,missing)


def solid_arrow():
    image = np.full((110,160,3),(91,65,60),np.uint8)
    cv2.line(image,(20,50),(100,50),(160,150,235),8)
    cv2.fillConvexPoly(image,np.array([[115,50],[95,40],[95,60]]),(160,150,235))
    return image


def test_multicolor_valid_single_arrow_is_not_split():
    image = solid_arrow()
    mask = foreground_mask(image)
    image[(mask>0)&(np.indices(mask.shape)[1]<60)] = (220,130,190)
    count,labels,stats,_ = arrow_components(image,foreground_mask(image))
    assert count==2 and len(stats)==2


def test_color_patch_without_independent_head_is_not_split():
    image = solid_arrow()
    cv2.line(image,(65,55),(65,95),(220,130,190),8)
    mask = foreground_mask(image)
    assert cv2.connectedComponents(mask)[0]==2
    count,labels,_,_ = arrow_components(image,mask)
    assert count==2
    assert np.array_equal(labels>0,mask>0)
