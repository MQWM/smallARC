from pathlib import Path
import cv2
import numpy as np
import pytest

from arrow_solver.vision import read_image,recognize,repair_small_color_gaps,find_head,RecognitionError
from image_assertions import ScreenVerifier
from arrow_solver.solver import solve


def test_level206_split_head_reconnects_and_verifies():
    image=read_image(Path(__file__).parent/'fixtures/recorded/level206.png')
    board=recognize(image).board
    assert not board.issues
    arrows=[a for a in board.arrows if a.cells==[(7,6),(8,6),(9,6),(9,5)]]
    assert len(arrows)==1
    arrow=arrows[0]
    assert arrow.direction=='UP'
    assert arrow.quality>.9
    assert len(board.arrows)==59
    assert len(solve(board).order)==59
    verifier=ScreenVerifier(image,board)
    assert verifier.matches(image,set())
    # Its single mask now contains the old head and body regions together.
    x,y,_,_=board.crop_rect
    assert verifier.masks[arrow.id][899-y,516-x]
    assert verifier.masks[arrow.id][946-y,516-x]
    assert verifier.masks[arrow.id][arrow.click_point[1]-y,arrow.click_point[0]-x]


@pytest.mark.parametrize('gap',[1,2])
def test_dim_seam_repaired_but_real_background_gap_kept(gap):
    background=(91,65,60)
    color=(232,153,160)
    image=np.full((30,30,3),background,np.uint8)
    image[5:25,10:20]=color
    mask=np.zeros((30,30),np.uint8)
    mask[5:25,10:20]=255
    mask[14:14+gap,10:20]=0
    image[14:14+gap,10:20]=(176,116,121)
    assert np.all(repair_small_color_gaps(image,mask)[14:14+gap,10:20])
    image[14:14+gap,10:20]=background
    assert not repair_small_color_gaps(image,mask)[14:14+gap,10:20].any()
    image[14:14+gap,10:20]=(20,140,250)  # unrelated ink must not bridge purple
    assert not repair_small_color_gaps(image,mask)[14:14+gap,10:20].any()


def test_round_tail_is_not_a_reliable_head():
    mask=np.zeros((40,120),np.uint8)
    cv2.line(mask,(8,28),(108,28),255,12)
    cv2.line(mask,(108,28),(108,8),255,12)
    with pytest.raises(RecognitionError):
        find_head(mask,14)
