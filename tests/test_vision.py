from pathlib import Path

import cv2
import numpy as np
import pytest

from image_assertions import ScreenVerifier
from arrow_solver.models import Board
from arrow_solver.solver import solve
from path_helpers import expand_vertices
from arrow_solver.vision import recognize, read_image, thin, suggest_crop

SAMPLE = Path(__file__).parent / "fixtures" / "level2.jpg"


@pytest.fixture(scope="module")
def sample():
    image = read_image(SAMPLE)
    return image, recognize(image).board


def test_real_board_has_complete_instances_and_plan(sample, tmp_path):
    image, board = sample
    assert len(board.arrows) == 59
    assert (board.cols, board.rows) == (20, 31)
    assert not board.issues
    plan = solve(board)
    assert len(plan.order) == 59
    # Each click identifies a separate foreground component; sampled paths fit it.
    verifier = ScreenVerifier(image, board)
    assert len(verifier.masks) == 59
    assert verifier.matches(image, set())
    board.save(tmp_path / "board.json")
    assert Board.load(tmp_path / "board.json").fingerprint() == board.fingerprint()


@pytest.mark.parametrize("scale", [.5, .75, 1.25])
def test_resized_image_preserves_all_paths(sample, scale):
    image, reference = sample
    smaller = recognize(cv2.resize(image, None, fx=scale, fy=scale)).board
    assert not smaller.issues
    signature = lambda b: {(tuple(a.cells), a.direction) for a in b.arrows}
    assert signature(smaller) == signature(reference)
    assert len(solve(smaller).order) == 59


def test_visually_checked_paths(sample):
    _, board = sample
    # Independently readable corner references, checked against the source/overlay.
    expected = [
        ("RIGHT", [[16, 2], [18, 2], [18, 0], [19, 0]]),
        ("DOWN", [[10, 4], [10, 2], [9, 2], [9, 7]]),
        ("DOWN", [[5, 28], [5, 29], [4, 29], [4, 27], [6, 27], [6, 28], [7, 28], [7, 29]]),
        ("RIGHT", [[17, 30], [16, 30], [16, 29], [17, 29], [17, 28], [18, 28], [18, 30],
                   [19, 30], [19, 27], [16, 27], [16, 28], [14, 28], [14, 26], [19, 26]]),
    ]
    for direction, corners in expected:
        assert any(a.direction == direction and a.cells == expand_vertices(corners) for a in board.arrows)


def test_thinning_keeps_bent_path_connected():
    mask = np.zeros((80, 100), np.uint8)
    cv2.polylines(mask, [np.array([[15, 60], [15, 20], [75, 20], [75, 60]])], False, 255, 9)
    skeleton = thin(mask)
    count, _ = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    assert count == 2
    assert skeleton.sum() < (mask > 0).sum() / 3


def test_incomplete_crop_reports_boundary(sample):
    image, board = sample
    x, y, w, h = board.crop_rect
    clipped = recognize(image, (x + 45, y, w - 45, h)).board
    assert any("边界" in issue for issue in clipped.issues)



def test_tracing_recovers_eroded_diagonal_tail():
    from arrow_solver.vision import trace, path_covers_body
    mask = np.zeros((145,160),np.uint8)
    cv2.polylines(mask,[np.array([[15,15],[65,65],[65,115],[100,115]])],False,255,8)
    cv2.line(mask,(100,115),(125,115),255,8)
    cv2.fillConvexPoly(mask,np.array([[140,115],[122,105],[122,125]]),255)
    path = trace(mask,np.array([140.,115.]),'RIGHT',8.)
    assert np.linalg.norm(path[0]-[15,15]) < 8
    assert path_covers_body(mask,path,np.array([140.,115.]),8.)
    assert not path_covers_body(mask,path[path[:,1]>65],np.array([140.,115.]),8.)


@pytest.mark.parametrize("scale", [.75, 1., 1.25])
def test_347_crop_contains_all_groups_at_different_scales(scale):
    image = read_image(Path(__file__).parent / "fixtures/recorded/level347.png")
    image = cv2.resize(image, None, fx=scale, fy=scale)
    x, y, w, h = suggest_crop(image)
    assert x < 248 * scale and x + w > 1037 * scale
    assert y < 741 * scale and y + h > 1708 * scale


def test_347_disconnected_groups_are_all_recognized():
    from arrow_solver.executor import ScreenValidator
    image = read_image(Path(__file__).parent / "fixtures/recorded/level347.png")
    board = recognize(image).board
    assert not board.issues
    assert len(board.arrows) == 56
    assert sum(a.click_point[1] < 1207 for a in board.arrows) == 10
    assert len(solve(board).order) == 56
    ScreenValidator(image, board)


def test_crop_keeps_thin_isolated_objects_and_excludes_ui_and_grid():
    image = np.full((1000, 500, 3), (80, 65, 60), np.uint8)
    # Broad UI regions must not outvote a sparse board.
    cv2.rectangle(image, (10, 20), (490, 130), (240, 240, 240), -1)
    cv2.rectangle(image, (10, 920), (490, 980), (240, 240, 240), -1)
    for y in range(200, 850, 20):
        cv2.circle(image, (30, y), 1, (180, 180, 180), -1)
    # Only three foreground pixels in each row: the old density cutoff loses it.
    cv2.line(image, (150, 250), (150, 290), (100, 220, 240), 2)
    cv2.line(image, (250, 600), (350, 800), (240, 180, 100), 4)
    # Unknown objects stay in the crop so recognition can report them.
    cv2.rectangle(image, (430, 400), (445, 415), (150, 220, 150), -1)
    x, y, w, h = suggest_crop(image)
    assert 30 < x < 149 and 150 <= y < 249
    assert x + w > 445 and 802 < y + h <= 900
