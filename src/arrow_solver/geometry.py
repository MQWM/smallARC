"""Continuous collision checks for paths which cannot be represented by a grid."""
from __future__ import annotations

from math import hypot, isfinite

from .models import DIRECTIONS


def point_distance(p, a, b):
    dx,dy = b[0]-a[0],b[1]-a[1]
    length = dx*dx+dy*dy
    t = max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)) if length else 0.
    return hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def segment_distance(a,b,c,d):
    """Exact segment distance, including crossings, parallel and point segments."""
    ux,uy = b[0]-a[0],b[1]-a[1]
    vx,vy = d[0]-c[0],d[1]-c[1]
    wx,wy = c[0]-a[0],c[1]-a[1]
    cross = ux*vy-uy*vx
    if abs(cross) > 1e-9:
        t = (wx*vy-wy*vx)/cross
        s = (wx*uy-wy*ux)/cross
        if 0 <= t <= 1 and 0 <= s <= 1:
            return 0.
    return min(point_distance(a,c,d),point_distance(b,c,d),
               point_distance(c,a,b),point_distance(d,a,b))


def rectangle_segments(rect):
    x,y,w,h = rect
    corners = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    return list(zip(corners,corners[1:]+corners[:1]))


def body_segments(board,arrow):
    points = arrow.pixel_path or [board.pixel(c) for c in arrow.cells]
    if arrow.kind == 'wide':
        # Its reserved lateral lanes are all solid, not just the centerline.
        return [(board.pixel(c),board.pixel(c)) for c in arrow.footprint], min(board.grid_step)*.5
    segments = list(zip(points,points[1:])) or [(points[0],points[0])]
    if arrow.kind == 'timer' and arrow.head_rect:
        segments += rectangle_segments(arrow.head_rect)
    return segments,board.line_width*.5


def launch_segments(board,arrow):
    """Extend all candidate directions to beyond the crop, stopping at portals."""
    vectors = arrow.launch_vectors or [DIRECTIONS[arrow.direction]]
    starts = [arrow.pixel_path[-1] if arrow.pixel_path else board.pixel(arrow.head)]
    radius = board.line_width*.5
    if arrow.kind == 'wide':
        starts = [board.pixel(c) for c in arrow.footprint]
        radius = min(board.grid_step)*.5
    length = hypot(board.image_width,board.image_height)*2
    for vector in vectors:
        norm = hypot(*vector)
        dx,dy = vector[0]/norm,vector[1]/norm
        center = arrow.pixel_path[-1] if arrow.pixel_path else board.pixel(arrow.head)
        entrances = []
        for portal in board.portals:
            px,py = portal.center[0]-center[0],portal.center[1]-center[1]
            if px*dx+py*dy > 0 and abs(px*dy-py*dx) <= board.line_width*.5:
                entrances.append(portal.center)
        for start in starts:
            stop = length
            for px,py in entrances:
                # Broad lateral lanes stop at the same central entrance plane.
                stop = min(stop,max(0.,(px-start[0])*dx+(py-start[1])*dy))
            yield start,(start[0]+dx*stop,start[1]+dy*stop),radius


def blocks(board,moving,obstacle):
    bodies,radius = body_segments(board,obstacle)
    return any(segment_distance(start,end,a,b) <= moving_radius+radius
               for start,end,moving_radius in launch_segments(board,moving)
               for a,b in bodies)


def counter_threshold(board,arrow):
    threshold = 0
    for block in board.counter_blocks:
        x,y,w,h = block.visual_rect
        edges = rectangle_segments(block.visual_rect)
        for start,end,radius in launch_segments(board,arrow):
            inside = x <= start[0] <= x+w and y <= start[1] <= y+h
            if inside or any(segment_distance(start,end,a,b) <= radius for a,b in edges):
                threshold = max(threshold,block.remaining)
                break
    return threshold


def validate_geometry(board,arrow):
    if len(arrow.pixel_path) < 2:
        raise ValueError('真实折线路径缺失')
    x,y,w,h = board.crop_rect
    for point in arrow.pixel_path:
        if len(point) != 2 or not all(isfinite(v) for v in point):
            raise ValueError('真实路径坐标无效')
        if not (x <= point[0] <= x+w and y <= point[1] <= y+h):
            raise ValueError('真实路径超出裁剪范围')
    for a,b in zip(arrow.pixel_path,arrow.pixel_path[1:]):
        if hypot(b[0]-a[0],b[1]-a[1]) < 1e-6:
            raise ValueError('真实路径包含重复点')
    direction = DIRECTIONS[arrow.direction]
    for vector in arrow.launch_vectors:
        if len(vector) != 2 or not all(isfinite(v) for v in vector) or hypot(*vector) < 1e-6:
            raise ValueError('发射方向向量无效')
        alignment = sum(v*d for v,d in zip(vector,direction))/(hypot(*vector)*hypot(*direction))
        if alignment < .94:
            raise ValueError('发射向量与头部方向不一致')
