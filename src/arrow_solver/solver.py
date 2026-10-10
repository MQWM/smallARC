from __future__ import annotations

import heapq

from .models import Board, DIRECTIONS, Plan
from . import geometry


class BoardError(ValueError):
    pass


def occupied_cells(arrow):
    if arrow.launch_vectors:
        # Quantized cells are retained for UI location, never for collisions.
        return set()
    cells = set(arrow.footprint or arrow.cells)
    if not arrow.footprint:
        for a,b in zip(arrow.cells,arrow.cells[1:]):
            if a[0]!=b[0] and a[1]!=b[1]:
                cells.add(((a[0]+b[0])/2,(a[1]+b[1])/2))
    return cells


def occupancy(board: Board) -> dict[tuple[int, int], int]:
    if not board.arrows:
        raise BoardError("没有识别到箭头")
    if board.cols <= 0 or board.rows <= 0 or min(board.grid_step) <= 0:
        raise BoardError("网格参数无效")
    used: dict[tuple[int, int], int] = {}
    ids = set()
    for arrow in board.arrows:
        if arrow.id in ids:
            raise BoardError(f"箭头编号重复：{arrow.id}")
        ids.add(arrow.id)
        if arrow.kind not in ("normal", "timer", "wide", "electric"):
            raise BoardError(f"箭头 {arrow.id} 类型未知")
        if not arrow.cells or arrow.direction not in DIRECTIONS:
            raise BoardError(f"箭头 {arrow.id} 的路径或方向无效")
        if len(set(arrow.cells)) != len(arrow.cells):
            raise BoardError(f"箭头 {arrow.id} 路径重复经过同一单元")
        for prev, cur in zip(arrow.cells, arrow.cells[1:]):
            if max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1])) != 1:
                raise BoardError(f"箭头 {arrow.id} 路径不连续")
        if arrow.launch_vectors:
            try:
                geometry.validate_geometry(board,arrow)
            except ValueError as exc:
                raise BoardError(f"箭头 {arrow.id}：{exc}") from exc
        elif len(arrow.cells) > 1:
            before, head = arrow.cells[-2:]
            if (head[0] - before[0], head[1] - before[1]) != DIRECTIONS[arrow.direction]:
                raise BoardError(f"箭头 {arrow.id} 的头部方向与末段路径不一致")
        if arrow.kind == "wide" and not arrow.footprint:
            raise BoardError(f"粗箭头 {arrow.id} 缺少占用范围")
        for cell in occupied_cells(arrow):
            if not board.inside(cell):
                raise BoardError(f"箭头 {arrow.id} 超出棋盘：{cell}")
            if cell in used and used[cell] != arrow.id:
                raise BoardError(f"箭头 {arrow.id} 与 {used[cell]} 重叠：{cell}")
            used[cell] = arrow.id
    for portal in board.portals:
        if not board.inside(portal.cell) or portal.cell in used:
            raise BoardError(f"黑瞳位置越界或与箭头重叠：{portal.cell}")
    for block in board.counter_blocks:
        if block.remaining is None or block.remaining < 1 or not board.inside(block.cell) or block.cell in used:
            raise BoardError(f"数字方块数据无效或位置重叠：{block.cell}")
    return used


def ray(board: Board, arrow):
    portals = {p.cell for p in board.portals}
    if arrow.kind == "timer" and arrow.footprint and arrow.head not in arrow.footprint:
        # The rounded nose cell lies ahead of the physical sprite, so it is an
        # exit blocker to check rather than part of the missile's own occupancy.
        if arrow.head in portals:
            return
        yield arrow.head
    if arrow.kind == "wide":
        dx, dy = DIRECTIONS[arrow.direction]
        seen = set()
        # A broad arrow enters when its central head reaches the portal;
        # all lateral lanes must be clear up to that same entrance plane.
        hx,hy = arrow.head
        stops = [(px-hx)*dx+(py-hy)*dy for px,py in portals
                 if (px-hx)*dy == (py-hy)*dx and (px-hx)*dx+(py-hy)*dy > 0]
        stop = min(stops) if stops else None
        for x, y in arrow.footprint:
            x, y = x+dx, y+dy
            while board.inside((x,y)):
                if stop is not None and (x-hx)*dx+(y-hy)*dy >= stop:
                    break
                if (x,y) not in seen:
                    seen.add((x,y))
                    yield x,y
                x, y = x+dx,y+dy
        return
    x, y = arrow.head
    dx, dy = DIRECTIONS[arrow.direction]
    dx,dy=dx*.5,dy*.5
    x, y = x + dx, y + dy
    while board.inside((x, y)):
        if (x,y) in portals:
            break
        yield x, y
        x, y = x + dx, y + dy


def block_threshold(board, arrow):
    if arrow.launch_vectors:
        return geometry.counter_threshold(board,arrow)
    return max((block.remaining for block in board.counter_blocks
                if any(max(abs(c[0]-block.cell[0]),abs(c[1]-block.cell[1])) <= .5
                       for c in ray(board,arrow))), default=0)


def dependencies(board,used):
    blockers = {a.id: sorted({used[c] for c in ray(board, a) if c in used and used[c] != a.id})
                for a in board.arrows}
    if not any(a.launch_vectors for a in board.arrows):
        return blockers
    for a in board.arrows:
        if a.launch_vectors:
            blockers[a.id] = []
        hits = set(blockers[a.id])
        for other in board.arrows:
            if a.id != other.id and (a.launch_vectors or other.launch_vectors) and geometry.blocks(board,a,other):
                hits.add(other.id)
        blockers[a.id] = sorted(hits)
    return blockers


def solve(board: Board) -> Plan:
    if board.turn_channels:
        from .channels import solve as solve_channels
        plan = solve_channels(board)
        verify_plan(board,plan)
        return plan
    used = occupancy(board)
    blockers = dependencies(board,used)
    dependents = {a.id: [] for a in board.arrows}
    degree = {a: len(bs) for a, bs in blockers.items()}
    for a, bs in blockers.items():
        for b in bs:
            dependents[b].append(a)
    timers = {a.id for a in board.arrows if a.kind == "timer"}
    urgent = set()
    pending = list(timers)
    while pending:
        current = pending.pop()
        for dependency in blockers[current]:
            if dependency not in urgent:
                urgent.add(dependency)
                pending.append(dependency)
    thresholds = {a.id:block_threshold(board,a) for a in board.arrows}
    def priority(aid):
        # Keep missile clearing first, but use currently free exits before
        # newly unlocked numbered lanes instead of rushing their count boundary.
        return (0 if aid in urgent else 1 if aid in timers else 2, thresholds[aid], aid)
    ready = [priority(a) for a, count in degree.items() if count == 0]
    heapq.heapify(ready)
    order = []
    # A fresh countdown starts on the first screen tap. If a timer already has
    # a clear exit, prefer any currently legal normal tap before waiting for it.
    normals = [entry for entry in ready if entry[-1] not in timers]
    if ready and ready[0][-1] in timers and normals:
        first = min(normals)
        ready.remove(first)
        heapq.heapify(ready)
        heapq.heappush(ready, (-1, first[1], first[-1]))
    successful_clicks = 0
    while ready:
        eligible = [entry for entry in ready if thresholds[entry[-1]] <= successful_clicks]
        if not eligible:
            raise BoardError(f"数字障碍尚未解锁且没有可消除箭头：已消除 {successful_clicks}，请检查数字、路径或特殊规则")
        chosen = min(eligible)
        ready.remove(chosen)
        heapq.heapify(ready)
        _, _, current = chosen
        if current not in timers:
            successful_clicks += 1
        order.append(current)
        for dependent in sorted(dependents[current]):
            degree[dependent] -= 1
            if degree[dependent] == 0:
                heapq.heappush(ready, priority(dependent))
    if len(order) != len(board.arrows):
        # Report a concrete cycle, not every downstream blocked node.
        remaining = {a for a, d in degree.items() if d}
        trail, positions = [], {}
        node = min(remaining)
        while node not in positions:
            positions[node] = len(trail)
            trail.append(node)
            node = next(b for b in blockers[node] if b in remaining)
        cycle = trail[positions[node]:] + [node]
        raise BoardError("阻挡依赖成环，请检查这些箭头的方向/路径：" + " → ".join(map(str, cycle)))
    plan = Plan(order, blockers, board.fingerprint())
    verify_plan(board, plan)
    return plan


def verify_plan(board: Board, plan: Plan) -> None:
    if board.turn_channels:
        from .channels import verify
        verify(board,plan)
        return
    used = occupancy(board)
    blockers = dependencies(board,used)
    arrows = {a.id: a for a in board.arrows}
    if len(plan.order) != len(arrows) or set(plan.order) != set(arrows):
        raise BoardError("计划必须包含每个箭头且仅一次")
    if plan.board_fingerprint != board.fingerprint():
        raise BoardError("棋盘已修改，执行计划失效")
    successful_clicks = 0
    remaining = set(arrows)
    for aid in plan.order:
        a = arrows[aid]
        if block_threshold(board,a) > successful_clicks:
            raise BoardError(f"计划非法：箭头 {aid} 前方数字方块尚未解锁")
        if any(other in remaining for other in blockers[aid]):
            raise BoardError(f"计划非法：箭头 {aid} 仍被阻挡")
        for c in occupied_cells(a):
            del used[c]
        remaining.remove(aid)
        if a.kind != "timer":
            successful_clicks += 1
    if used:
        raise BoardError("模拟执行后棋盘未清空")
