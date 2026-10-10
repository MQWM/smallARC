import pytest

from arrow_solver.ui.path_editor import edited_arrow
from arrow_solver.solver import solve
from test_executor import scene


def test_drawn_bends_expand_to_cells_and_infer_direction():
    _,board = scene()
    arrow = edited_arrow(board,[(20,100),(100,100),(100,140)])
    assert arrow.cells == [(0,2),(1,2),(2,2),(2,3)]
    assert arrow.direction == 'DOWN'
    board.arrows.append(arrow)
    assert len(solve(board).order)==3


def test_non45_drawn_path_keeps_real_line_and_both_directions():
    _,board = scene()
    arrow = edited_arrow(board,[(20,100),(180,140)])
    assert arrow.pixel_path == [(20,100),(180,140)]
    assert len(arrow.launch_vectors)==2
    assert arrow.direction=='RIGHT'
    board.arrows.append(arrow)
    assert len(solve(board).order)==3


def test_click_point_only_edit_preserves_existing_geometry():
    _,board = scene()
    current = board.arrows[0]
    current.pixel_path = [(60.,60.),(100.,60.)]
    result = edited_arrow(board,current.pixel_path,current,click=(95,60),direction='RIGHT')
    assert result.cells==current.cells
    assert result.pixel_path==current.pixel_path
    assert result.click_point==(95,60)
    assert current.click_point==(100,60)


def test_invalid_graphic_paths_are_rejected():
    _,board = scene()
    with pytest.raises(ValueError,match='至少'):
        edited_arrow(board,[(20,100)])
    with pytest.raises(ValueError,match='末段'):
        edited_arrow(board,[(20,100),(100,100)],direction='UP')
    with pytest.raises(ValueError,match='超出'):
        edited_arrow(board,[(-100,100),(100,100)])
