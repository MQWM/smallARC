import random

import pytest

from arrow_solver.models import Arrow, Board, Plan
from arrow_solver.solver import BoardError, solve, verify_plan


def board(arrows, cols=8, rows=8):
    return Board(400, 400, (0, 0, 400, 400), (20, 20), (40, 40), cols, rows, arrows)


def arrow(aid, cells, direction="RIGHT"):
    return Arrow(aid, cells, direction, (20, 20))


def test_all_blockers_and_duplicate_hits():
    b = board([arrow(1, [(0, 1), (1, 1)]),
               arrow(2, [(3, 1), (4, 1)]),
               arrow(3, [(6, 2), (6, 1), (6, 0)], "UP")])
    p = solve(b)
    assert p.blockers[1] == [2, 3]
    assert p.order.index(3) < p.order.index(2) < p.order.index(1)


def test_self_in_ray_is_ignored():
    a = arrow(1, [(4, 1), (4, 2), (3, 2), (2, 2), (2, 1), (3, 1)])
    p = solve(board([a]))
    assert p.order == [1]
    assert p.blockers == {1: []}


def test_dependency_cycle_is_reported():
    b = board([arrow(1, [(0, 1), (1, 1)]), arrow(2, [(4, 1), (3, 1)], "LEFT")])
    with pytest.raises(BoardError, match="成环"):
        solve(b)


@pytest.mark.parametrize("arrows, message", [
    ([arrow(1, [(0, 1), (2, 1)])], "不连续"),
    ([arrow(1, [(0, 1), (1, 1)], "UP")], "不一致"),
    ([arrow(1, [(1, 1)]), arrow(2, [(1, 1)])], "重叠"),
    ([arrow(1, [(-1, 1)])], "超出"),
    ([arrow(1, [(1, 1)]), arrow(1, [(2, 1)])], "重复"),
])
def test_invalid_models(arrows, message):
    with pytest.raises(BoardError, match=message):
        solve(board(arrows))


def test_stale_and_incorrect_plans():
    b = board([arrow(1, [(0, 1)]), arrow(2, [(3, 1)])])
    plan = solve(b)
    with pytest.raises(BoardError, match="仍被阻挡"):
        verify_plan(b, Plan([1, 2], {}, b.fingerprint()))
    b.revision += 1
    with pytest.raises(BoardError, match="失效"):
        verify_plan(b, plan)


def test_randomized_grid_order_independent():
    rng = random.Random(54)
    for _ in range(30):
        cells = rng.sample([(x, y) for x in range(8) for y in range(8)], 25)
        arrows = [arrow(i + 1, [c]) for i, c in enumerate(cells)]
        rng.shuffle(arrows)
        b = board(arrows)
        p = solve(b)
        # Independent check: each clicked row must have no remaining cell to its right.
        remaining = {a.id: a.head for a in arrows}
        for aid in p.order:
            x, y = remaining.pop(aid)
            assert all(oy != y or ox < x for ox, oy in remaining.values())



def test_clear_timer_gets_a_legal_normal_tap_to_start_clock():
    timer = arrow(1, [(0,0),(1,0)])
    timer.kind = "timer"
    b = board([timer, arrow(2,[(0,2),(1,2)])])
    assert solve(b).order == [2,1]
