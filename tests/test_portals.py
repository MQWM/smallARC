from copy import deepcopy
from pathlib import Path
import json

import cv2
import pytest

from arrow_solver.models import Arrow,Board,Portal
from arrow_solver.vision import read_image,recognize
from arrow_solver.solver import solve,BoardError,verify_plan
from image_assertions import ScreenVerifier
from arrow_solver.renderer import export_run


def board(arrows,portals):
    return Board(200,200,(0,0,200,200),(0,0),(20,20),10,10,arrows,portals=portals)


def portal(cell):
    return Portal(cell,(cell[0]*20,cell[1]*20),(0,0,10,10))


def arrow(aid,cells,direction='RIGHT'):
    return Arrow(aid,cells,direction,(0,0))


def test_portal_breaks_cycle_and_is_reusable_in_both_directions():
    b=board([arrow(1,[(0,2),(1,2)]),arrow(2,[(6,2),(5,2)],'LEFT')],[])
    with pytest.raises(BoardError,match='成环'):
        solve(b)
    b.portals=[portal((3,2))]
    plan=solve(b)
    assert plan.blockers=={1:[],2:[]}
    assert plan.order==[1,2]
    verify_plan(b,plan)
    assert len(b.portals)==1


def test_only_blockers_before_aligned_portal_count():
    b=board([arrow(1,[(0,2),(1,2)]),arrow(2,[(2,2),(2,1)],'UP'),
             arrow(3,[(6,2),(7,2)])],[portal((4,2))])
    assert solve(b).blockers[1]==[2]
    b.portals=[portal((4,3))]
    assert solve(b).blockers[1]==[2,3]


def test_wide_exit_checks_side_lane_until_portal_plane():
    broad=arrow(1,[(2,0),(2,1)],'DOWN')
    broad.kind='wide'
    broad.footprint=[(x,y) for x in (1,2,3) for y in (0,1)]
    b=board([broad,arrow(2,[(3,4),(4,4)]),arrow(3,[(3,7),(4,7)])],[portal((2,5))])
    assert solve(b).blockers[1]==[2]


@pytest.fixture(scope='module')
def scene():
    root=Path(__file__).parent/'fixtures/recorded'
    image=read_image(root/'level203.png')
    return image,recognize(image).board,read_image(root/'level203-rotated.png')


def test_real_portals_and_roundtrip_export(scene,tmp_path):
    image,b,_=scene
    assert not b.issues
    assert len(b.arrows)==122
    assert {p.cell for p in b.portals}=={(11,7),(18,16),(12,19),(15,30)}
    plan=solve(b)
    assert len(plan.order)==122
    path=export_run(tmp_path,image,b,plan)
    restored=Board.load(path/'board.json')
    assert restored.fingerprint()==b.fingerprint()
    data=json.loads((path/'solution.json').read_text())
    assert len(data['portals'])==4
    assert len(data['actions'])==len(data['clicks'])==122
    without=deepcopy(b)
    without.portals=[]
    # Ignoring these entrances creates a false dependency cycle in this real level.
    with pytest.raises(BoardError,match="成环"):
        solve(without)


def test_real_rotation_is_allowed_but_lost_portal_and_wrong_arrow_are_not(scene):
    image,b,rotated=scene
    verifier=ScreenVerifier(image,b)
    assert verifier.matches(rotated,set())
    missing=rotated.copy()
    x,y,w,h=b.portals[0].visual_rect
    missing[y:y+h,x:x+w]=verifier.background
    assert not verifier.matches(missing,set())
    aid=solve(b).order[0]
    after=rotated.copy()
    after[verifier.roi][verifier.masks[aid]]=verifier.background
    assert verifier.matches(after,{aid})
    assert not verifier.matches(after,set())
    assert not verifier.matches(rotated,{aid})


@pytest.mark.parametrize('scale',[.5,.75,1.25])
def test_portal_locations_scale_with_grid(scene,scale):
    image,b,_=scene
    other=recognize(cv2.resize(image,None,fx=scale,fy=scale)).board
    assert {p.cell for p in other.portals}=={p.cell for p in b.portals}
