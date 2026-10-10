"""Convert image clicks to an arrow without requiring coordinate entry."""
from copy import deepcopy

import numpy as np

from ..models import Arrow, DIRECTIONS


def snapped_point(board, point):
    center = board.pixel(board.cell(point))
    return center if np.linalg.norm(np.array(center)-point) <= min(board.grid_step)*.22 else tuple(point)


def edited_arrow(board, points, current=None, click=None, direction=None):
    if len(points) < 2:
        raise ValueError('请至少点击尾部和头部两个点')
    points = [tuple(map(float,p)) for p in points]
    if any(not (0 <= p[0] < board.image_width and 0 <= p[1] < board.image_height) for p in points):
        raise ValueError('路径超出截图')
    original = current.pixel_path or [board.pixel(c) for c in current.cells] if current else []
    if current and points == original and (direction is None or direction == current.direction):
        result = deepcopy(current)
        if click is not None:
            result.click_point = tuple(map(round,click))
        return result
    if current and current.kind in ('timer','wide'):
        raise ValueError('导弹和粗箭头需保留完整视觉占用；可点选调整点击点，路径请重新识别')
    points = [snapped_point(board,p) for p in points]
    vectors = np.diff(points,axis=0)
    if np.any(np.linalg.norm(vectors,axis=1) < board.line_width*.5):
        raise ValueError('相邻点太近，请撤销重复点')
    terminal = vectors[-1]/np.linalg.norm(vectors[-1])
    direction = direction or max(DIRECTIONS,key=lambda d: np.dot(terminal,np.array(DIRECTIONS[d])/np.linalg.norm(DIRECTIONS[d])))
    vector = np.array(DIRECTIONS[direction],dtype=float)
    vector /= np.linalg.norm(vector)
    if np.dot(terminal,vector) < .94:
        raise ValueError('头部方向与末段不一致，请调整末段或方向')
    corners = [board.cell(p) for p in points]
    skew = any(a[0]!=b[0] and a[1]!=b[1] and abs(a[0]-b[0])!=abs(a[1]-b[1]) for a,b in zip(corners,corners[1:]))
    skew |= any(min(abs(v)) > board.line_width and abs(abs(v[0])-abs(v[1])) >= board.line_width for v in vectors)
    skew |= any(np.linalg.norm(np.array(board.pixel(c))-p) > board.line_width*.5 for c,p in zip(corners,points))
    cells = []
    for a,b in zip(corners,corners[1:]):
        for p in np.linspace(a,b,max(2,max(abs(b[i]-a[i]) for i in range(2))*2+1)):
            cell = tuple(map(round,p))
            if not cells or cells[-1] != cell:
                cells.append(cell)
    if any(not board.inside(c) for c in cells):
        raise ValueError('路径超出识别到的棋盘范围')
    if len(set(cells)) != len(cells):
        raise ValueError('路径自交或重复，请调整控制点')
    launch = [tuple(vector)] if skew else []
    if skew and not np.allclose(terminal,vector,atol=.03):
        launch.append(tuple(terminal))
    return Arrow(current.id if current else max((a.id for a in board.arrows),default=0)+1,
                 cells,direction,tuple(map(round,click or points[-1])),
                 color=current.color if current else (255,200,80),kind=current.kind if current else 'normal',
                 visual_rect=current.visual_rect if current else None,
                 pixel_path=points,launch_vectors=launch)
