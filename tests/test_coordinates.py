from arrow_solver.models import Arrow, Board
from path_helpers import expand_vertices
import pytest


def test_coordinates_and_json_roundtrip(tmp_path):
    b = Board(1200, 2600, (70, 450, 1000, 1700), (104.2, 620.1), (52.3, 52.4), 20, 30,
              [Arrow(1, [(2, 3), (3, 3)], "RIGHT", (261, 777))])
    for cell in [(0, 0), (2, 3), (19, 29)]:
        assert b.cell(b.pixel(cell)) == cell
    b.save(tmp_path / "board.json")
    other = Board.load(tmp_path / "board.json")
    assert b.fingerprint() == other.fingerprint()


def test_manual_vertices():
    assert expand_vertices([[0, 0], [2, 0], [2, 2]]) == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    with pytest.raises(ValueError, match="横向"):
        expand_vertices([[0, 0], [2, 3]])

