from pathlib import Path

import cv2
import numpy as np
import pytest

from image_assertions import ScreenVerifier
from arrow_solver.renderer import export_run
from arrow_solver.solver import solve, occupied_cells
from arrow_solver.vision import read_image, recognize

FIXTURES=Path(__file__).parent/'fixtures/recorded'


@pytest.fixture(scope='module')
def flame_scene():
    image=read_image(FIXTURES/'level222.png')
    board=recognize(image).board
    arrow=next(a for a in board.arrows if a.kind=='normal' and a.visual_rect)
    return image,board,arrow


def test_flame_skin_uses_normal_path_dependencies_and_tap(flame_scene,tmp_path):
    import json
    image,board,arrow=flame_scene
    assert not board.issues
    assert len(board.arrows)==117
    assert arrow.direction=='LEFT'
    assert arrow.head==(25,43)
    assert arrow.cells[0]==(31,49)
    assert len(arrow.cells)==41
    assert not arrow.footprint
    assert occupied_cells(arrow)==set(arrow.cells)
    plan=solve(board)
    assert len(plan.order)==117
    assert all(plan.order.index(aid)<plan.order.index(arrow.id) for aid in plan.blockers[arrow.id])
    verifier=ScreenVerifier(image,board)
    assert verifier.matches(image,set())
    x,y=arrow.click_point
    hsv=cv2.cvtColor(image[y:y+1,x:x+1],cv2.COLOR_BGR2HSV)[0,0]
    assert hsv[1]>150 and hsv[2]>160
    run=export_run(tmp_path,image,board,plan)
    action=next(a for a in json.loads((run/'solution.json').read_text())['actions'] if a['arrow']==arrow.id)
    assert action['action']=='tap'
    assert action['point']==list(arrow.click_point)


def test_flame_texture_can_animate_but_disappearance_still_requires_verification(flame_scene):
    image,board,arrow=flame_scene
    later=read_image(FIXTURES/'level222-later.png')
    verifier=ScreenVerifier(image,board)
    assert verifier.matches(later,set())
    next_board=recognize(later).board
    assert not next_board.issues
    next_arrow=next(a for a in next_board.arrows if a.kind=='normal' and a.visual_rect)
    assert next_arrow.cells==arrow.cells
    assert next_arrow.direction==arrow.direction
    gone=later.copy()
    gone[verifier.roi][verifier.masks[arrow.id]]=verifier.background
    assert verifier.matches(gone,{arrow.id})
    assert not verifier.matches(gone,set())
    assert not verifier.matches(later,{arrow.id})
    ordinary=next(a for a in board.arrows if a.visual_rect is None)
    extra=gone.copy()
    extra[verifier.roi][verifier.masks[ordinary.id]]=verifier.background
    assert not verifier.matches(extra,{arrow.id})


@pytest.mark.parametrize('scale',[.75,1.25])
def test_flame_path_scales_with_board(flame_scene,scale):
    image,_,reference=flame_scene
    board=recognize(cv2.resize(image,None,fx=scale,fy=scale)).board
    arrow=next(a for a in board.arrows if a.kind=='normal' and a.visual_rect)
    assert arrow.cells==reference.cells
    assert arrow.direction==reference.direction
