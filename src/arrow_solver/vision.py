from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .models import Arrow, Board, DIRECTIONS, image_hash


class RecognitionError(ValueError):
    pass


@dataclass
class Recognition:
    board: Board
    foreground: np.ndarray


def read_image(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RecognitionError("无法读取图片，请选择 PNG 或 JPEG")
    return image


def write_image(path, image) -> None:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RecognitionError("图片编码失败")
    data.tofile(str(path))


def foreground_mask(image: np.ndarray) -> np.ndarray:
    # Dominant quantized BGR bin is a robust estimate for the flat board background.
    small = image[::3, ::3].reshape(-1, 3)
    bins = small.astype(np.int32) // 8
    codes = bins[:, 0] * 1024 + bins[:, 1] * 32 + bins[:, 2]
    mode = np.bincount(codes).argmax()
    background = np.median(small[codes == mode], axis=0).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - bg_lab, axis=2)
    # Otsu separates flat background/JPEG noise from colored arrow interiors.
    _, mask = cv2.threshold(np.clip(distance, 0, 255).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return repair_small_color_gaps(image, mask)


def repair_small_color_gaps(image, mask):
    """Recover dim colored seams, without filling actual background gaps."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = mask.shape
    padded_mask = np.pad(mask, 3)
    padded_hsv = np.pad(hsv, ((3,3),(3,3),(0,0)))
    repaired = mask.copy()

    def shifted(dx, dy):
        return padded_mask[3+dy:3+dy+h, 3+dx:3+dx+w]

    def same_color(a, b):
        a,b = a.astype(np.int16),b.astype(np.int16)
        hue = np.abs(a[:,0]-b[:,0])
        return (np.minimum(hue,180-hue)<6) & (np.abs(a[:,1]-b[:,1])<25)

    # Inspect only missing pixels bracketed by foreground. This keeps color
    # comparisons sparse even for large phone screenshots.
    missing = mask == 0
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        for before,after in ((1,1),(1,2),(2,1)):
            candidates = missing & (shifted(-dx*before,-dy*before)>0) & (shifted(dx*after,dy*after)>0)
            yy,xx = np.nonzero(candidates)
            if not len(xx):
                continue
            left = padded_hsv[yy+3-dy*before,xx+3-dx*before]
            right = padded_hsv[yy+3+dy*after,xx+3+dx*after]
            current = hsv[yy,xx]
            valid = same_color(left,right) & same_color(current,left) & same_color(current,right)
            minimum = np.minimum(left[:,2],right[:,2])*.6
            valid &= current[:,2] >= minimum
            if before==2 or after==2:
                mx,my = (-dx,-dy) if before==2 else (dx,dy)
                middle = padded_hsv[yy+3+my,xx+3+mx]
                valid &= same_color(middle,left) & (middle[:,2]>=minimum)
            repaired[yy[valid],xx[valid]] = 255
    return repaired


def suggest_crop(image: np.ndarray) -> tuple[int, int, int, int]:
    """Include disconnected gameplay objects; user can override on the image."""
    h, w = image.shape[:2]
    from .orbit import detect
    orbit = detect(image)
    if orbit:
        x,y,bw,bh = orbit.track_rect
        return (round(x+4),round(y+4),round(bw-8),round(bh-8))
    # The header and toolbar sit outside this game area. Empty rows inside it
    # do not divide a board: e.g. level 347 has three separated arrow groups.
    area_top, area_bottom = round(h * .15), round(h * .9)
    mask = foreground_mask(image)[area_top:area_bottom]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    minimum_area = max(12, w * h * .000005)
    minimum_extent = max(5, w * .006)
    candidates = [i for i in range(1, len(stats))
                  if stats[i, 4] >= minimum_area
                  and max(stats[i, 2:4]) >= minimum_extent]
    if not candidates:
        raise RecognitionError("游戏区域没有可识别的前景")
    # Keep unknown sizable objects too; recognition must report them rather
    # than silently shrinking the crop around only the recognized arrows.
    objects = stats[candidates]
    left, top = objects[:, :2].min(axis=0)
    right, bottom = (objects[:, :2] + objects[:, 2:4]).max(axis=0)
    selected = np.isin(labels, candidates).astype(np.uint8)
    dt = cv2.distanceTransform(selected, cv2.DIST_L2, 5)
    maxima = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 1)
    line_width = 2 * float(np.median(dt[maxima])) if maxima.any() else 2.
    pad = max(6, round(w * .012), round(line_width * 2))
    left, right = max(0, int(left) - pad), min(w, int(right) + pad)
    top = max(area_top, area_top + int(top) - pad)
    bottom = min(area_bottom, area_top + int(bottom) + pad)
    return left, top, right - left, bottom - top


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang–Suen thinning. Returns bool, no contrib/OpenCV GUI dependency."""
    a = np.pad(mask > 0, 1).astype(np.uint8)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            neighbors = sum(p)
            transitions = sum((p[i] == 0) & (p[(i + 1) % 8] == 1) for i in range(8))
            if phase == 0:
                constraint = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                constraint = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            delete = (a[1:-1, 1:-1] > 0) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & constraint
            if np.any(delete):
                a[1:-1, 1:-1][delete] = 0
                changed = True
        if not changed:
            return a[1:-1, 1:-1].astype(bool)


def thin_guo_hall(mask: np.ndarray) -> np.ndarray:
    """Preserve diagonal endpoints that Zhang–Suen can erode away."""
    a = np.pad(mask > 0, 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            transitions = sum((~p[i]) & (p[(i+1) % 8] | p[(i+2) % 8]) for i in (0, 2, 4, 6))
            n1 = sum(p[i] | p[(i+1) % 8] for i in (7, 1, 3, 5))
            n2 = sum(p[i] | p[(i+1) % 8] for i in (0, 2, 4, 6))
            neighbors = np.minimum(n1, n2)
            protected = ((p[4] | p[5] | ~p[7]) & p[6]) if phase == 0 else ((p[0] | p[1] | ~p[3]) & p[2])
            delete = a[1:-1, 1:-1] & (transitions == 1) & (neighbors >= 2) & (neighbors <= 3) & ~protected
            if delete.any():
                a[1:-1, 1:-1][delete] = False
                changed = True
        if not changed:
            return a[1:-1, 1:-1]


def find_head(mask: np.ndarray, line_width: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    ambiguous = False
    for epsilon in (.28, .20, .36, .44):
        approx = cv2.approxPolyDP(contour, max(1., line_width * epsilon), True)[:, 0].astype(float)
        candidates = sorted(_head_candidates(mask, approx, line_width), key=lambda c: c[0], reverse=True)
        if not candidates:
            continue
        best = candidates[0]
        if best[0] < .35:
            continue
        if len(candidates) > 1 and candidates[1][0] > best[0] * .9:
            ambiguous = True
            continue
        return best[1], best[2], float(best[0])
    if ambiguous:
        raise RecognitionError("检测到多个可能的头部，需要拆分或人工确认")
    raise RecognitionError("未找到可靠的三角形头部")


def _head_candidates(mask, approx, line_width):
    candidates = []
    for i, tip in enumerate(approx):
        p, q = approx[i - 1], approx[(i + 1) % len(approx)]
        u, v = p - tip, q - tip
        lengths = np.linalg.norm(u), np.linalg.norm(v)
        if min(lengths) < line_width * .65:
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(u, v) / np.prod(lengths), -1, 1)))
        base = (p + q) / 2
        vector = tip - base
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        direction = max(DIRECTIONS, key=lambda d: np.dot(vector / norm, np.array(DIRECTIONS[d])/np.linalg.norm(DIRECTIONS[d])))
        alignment = np.dot(vector / norm, np.array(DIRECTIONS[direction])/np.linalg.norm(DIRECTIONS[direction]))
        width = np.linalg.norm(p - q)
        if not (25 < angle < 85 and alignment > .94 and width > line_width * 1.45):
            continue
        # The triangle midpoint must be foreground (reject concave corners).
        mid = np.rint((tip + base) / 2).astype(int)
        if not mask[mid[1], mid[0]]:
            continue
        score = alignment * min(lengths) / max(lengths) * min(width / (2 * line_width), 1.)
        candidates.append((score, tip, direction))
    return candidates


def trace(mask: np.ndarray, tip: np.ndarray, direction: str, line_width: float) -> np.ndarray:
    path = trace_chain(mask, thin(mask), tip, direction, line_width)
    if not path_covers_body(mask, path, tip, line_width):
        path = trace_chain(mask, thin_guo_hall(mask), tip, direction, line_width)
        if not path_covers_body(mask, path, tip, line_width):
            raise RecognitionError("路径未覆盖完整箭身，可能丢失斜向尾部")
    return path


def path_covers_body(mask, path, tip, line_width):
    distance_input = np.ones(mask.shape, np.uint8)
    points = np.rint(path).astype(int)
    distance_input[points[:, 1], points[:, 0]] = 0
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    # Exempt the actual triangle, including antialiased contour edges, rather
    # than a fixed radius which cuts through larger heads on small arrows.
    head = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    for epsilon in (.28, .20, .36, .44):
        vertices = cv2.approxPolyDP(contour, max(1., line_width*epsilon), True)[:, 0].astype(float)
        candidates = _head_candidates(mask, vertices, line_width)
        if not any(score >= .35 and np.linalg.norm(candidate-tip) < 1 for score,candidate,_ in candidates):
            continue
        index = int(np.argmin(np.linalg.norm(vertices-tip, axis=1)))
        triangle = np.rint([vertices[index-1], vertices[index], vertices[(index+1)%len(vertices)]]).astype(np.int32)
        cv2.fillConvexPoly(head, triangle, 255)
        head = cv2.dilate(head, np.ones((3,3),np.uint8))
        break
    yy, xx = np.nonzero((mask > 0) & (head == 0))
    # Test all body ink, not just skeleton pixels: thinning can erase a whole
    # branch while leaving a connected, apparently valid shortened skeleton.
    return not np.any(distance[yy, xx] > max(2., line_width*1.25))


def trace_chain(mask, skeleton, tip, direction, line_width):
    points = np.argwhere(skeleton)  # y,x
    if len(points) < 2:
        raise RecognitionError("箭头身体过短或分割失败")
    target = tip - np.array(DIRECTIONS[direction]) * line_width * .7
    start = tuple(points[np.argmin(np.sum((points[:, ::-1] - target) ** 2, axis=1))])
    pixels = set(map(tuple, points))
    previous = {start: None}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nxt = (cur[0] + dy, cur[1] + dx)
                if (dy or dx) and nxt in pixels and nxt not in previous:
                    previous[nxt] = cur
                    distances[nxt] = distances[cur] + 1
                    queue.append(nxt)
    if len(previous) != len(pixels):
        raise RecognitionError("骨架断裂")
    tail = max(distances, key=distances.get)
    path = []
    cur = tail
    while cur is not None:
        path.append((cur[1], cur[0]))
        cur = previous[cur]
    path.append(tuple(target))
    # Only short triangle branches are allowed outside the main chain.
    main_set = {(int(p[1]), int(p[0])) for p in path[:-1]}
    extras = pixels - main_set
    distance_input = np.ones(mask.shape, np.uint8)
    for p in main_set:
        distance_input[p] = 0
    distance_from_path = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
    if any(distance_from_path[p] > line_width * .8 and
           np.linalg.norm(np.array((p[1], p[0])) - tip) > line_width * 3.5 for p in extras):
        raise RecognitionError("身体存在异常分支，可能粘连或自交")
    return np.asarray(path, dtype=float)


def fit_grid(paths: list[np.ndarray], line_width: float):
    centers = [[], []]
    for path in paths:
        simplified = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), line_width * .5, False)[:, 0]
        for p, q in zip(simplified, simplified[1:]):
            delta = q - p
            if abs(delta[1]) > line_width * 2.5 and abs(delta[0]) < line_width:
                centers[0].append(float((p[0] + q[0]) / 2))
            if abs(delta[0]) > line_width * 2.5 and abs(delta[1]) < line_width:
                centers[1].append(float((p[1] + q[1]) / 2))
    fits = []
    for values in centers:
        groups = []
        for value in sorted(values):
            if not groups or value - np.mean(groups[-1]) > line_width:
                groups.append([value])
            else:
                groups[-1].append(value)
        coords = np.array([np.median(g) for g in groups])
        if len(coords) < 3:
            raise RecognitionError("网格线索不足，请手动设置网格原点、间距和范围")
        diffs = np.diff(coords)
        candidates = [d / k for d in diffs for k in range(1, 5) if line_width * 2 < d / k < line_width * 10]
        scored = []
        for step in candidates:
            origin = coords[0]
            indices = np.rint((coords - origin) / step)
            slope, intercept = np.polyfit(indices, coords, 1)
            residual = np.abs(coords - (intercept + indices * slope))
            if np.quantile(residual, .9) < line_width * .4:
                scored.append((slope, intercept, residual.mean()))
        if not scored:
            raise RecognitionError("网格间距拟合失败，请手动设置网格")
        # Largest well-supported lattice avoids choosing an unnecessary half-grid.
        step, origin, _ = max(scored, key=lambda v: v[0])
        fits.append((origin, step))
    return (fits[0][0], fits[1][0]), (fits[0][1], fits[1][1])


def recognize(image: np.ndarray, crop=None) -> Recognition:
    from .orbit import detect,mask_objects
    from .channels import find_channels
    from .special import find_timer_shape, read_countdown, find_wide_shape, find_portals, find_counter_blocks, flame_arrow_mask, arrow_components
    h, w = image.shape[:2]
    crop = tuple(crop or suggest_crop(image))
    x, y, cw, ch = map(int, crop)
    if not (0 <= x < w and 0 <= y < h and cw > 0 and ch > 0 and x + cw <= w and y + ch <= h):
        raise RecognitionError("裁剪范围超出图片")
    roi = image[y:y + ch, x:x + cw]
    portals = find_portals(image, crop)
    counter_blocks = find_counter_blocks(image,crop)
    channels = find_channels(image,crop)
    orbit = detect(image)
    foreground = mask_objects(foreground_mask(roi),crop,orbit)
    n, labels, stats, _ = arrow_components(roi, foreground)
    candidates = []
    widths = []
    for label in range(1, n):
        bx, by, bw, bh, area = stats[label]
        if area < max(12, cw * ch * .000015):
            continue
        if any(px <= x+bx and py <= y+by and x+bx+bw <= px+pw and y+by+bh <= py+ph
               for _, (px,py,pw,ph) in portals):
            continue
        if any(px-2 <= x+bx and py-2 <= y+by and x+bx+bw <= px+pw+2 and y+by+bh <= py+ph+2
               for _,(px,py,pw,ph),_ in counter_blocks):
            continue
        if any(px-2<=x+bx and py-2<=y+by and x+bx+bw<=px+pw+2 and y+by+bh<=py+ph+2
               for px,py,pw,ph in (c.visual_rect for c in channels)):
            continue
        mask = (labels[by:by + bh, bx:bx + bw] == label).astype(np.uint8) * 255
        mask = np.pad(mask, 3)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # Median of local medial maxima estimates stroke radius, resistant to triangle heads.
        maxima = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 1)
        width = 2 * float(np.median(dt[maxima])) if maxima.any() else 2.
        widths.append(width)
        candidates.append((mask, np.array((x + bx - 3, y + by - 3)), (bx, by, bw, bh)))
    if not candidates:
        raise RecognitionError("棋盘范围内未找到箭头")
    line_width = float(np.median(widths))
    paths, info, issues = [], [], []
    failures = []
    timer_heads = []
    wide_shapes = {}
    for index, (mask, offset, bbox) in enumerate(candidates, 1):
        shape = find_wide_shape(mask, line_width)
        if shape:
            wide_shapes[index] = (shape, offset)
    for index, (mask, offset, bbox) in enumerate(candidates, 1):
        bx, by, bw, bh = bbox
        # Decorative holes/fragments inside a recognized broad silhouette are not instances.
        inside_wide = False
        for other, (shape, outer_offset) in wide_shapes.items():
            if other == index:
                continue
            ys, xs = np.nonzero(mask)
            positions = np.column_stack((xs, ys)) + offset - outer_offset
            positions = positions.astype(int)
            filled = shape[3]
            valid = ((positions[:,0] >= 0) & (positions[:,0] < filled.shape[1]) &
                     (positions[:,1] >= 0) & (positions[:,1] < filled.shape[0]))
            if valid.all() and np.all(filled[positions[:,1], positions[:,0]]):
                inside_wide = True
                break
        if inside_wide:
            continue
        if bx == 0 or by == 0 or bx + bw == cw or by + bh == ch:
            issues.append(f"区域 {index} 接触裁剪边界，棋盘可能不完整，请扩大范围")
        if index in wide_shapes:
            (path, direction, click, filled), _ = wide_shapes[index]
            paths.append(path+offset)
            info.append((index, direction, .9, tuple(int(v) for v in click+offset),
                         (235,45,45), "wide", tuple(int(v) for v in (x+bx,y+by,bw,bh)), None))
            continue
        try:
            flame = flame_arrow_mask(image,(x+bx,y+by,bw,bh),mask)
            path_mask,path_width = flame if flame else (mask,line_width)
            tip, direction, quality = find_head(path_mask, path_width)
            path = trace(path_mask, tip, direction, path_width) + offset
            paths.append(path)
            click = tip - np.array(DIRECTIONS[direction]) * path_width * .7 + offset
            if flame:
                # Click visible skin near the head rather than a decorative hole.
                patch=image[y+by:y+by+bh,x+bx:x+bx+bw]
                hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
                yy,xx=np.nonzero((mask[3:-3,3:-3]>0)&(hsv[:,:,1]>150)&(hsv[:,:,2]>160))
                points=np.column_stack((xx+x+bx,yy+y+by))
                click=points[np.argmin(np.sum((points-click)**2,axis=1))]
            bx, by, bw, bh = bbox
            color_pixels = roi[by:by + bh, bx:bx + bw][mask[3:-3, 3:-3] > 0]
            color = tuple(int(c) for c in np.median(color_pixels, axis=0)[::-1])
            pixels=color_pixels.astype(float)
            spark=((pixels[:,0]>200)&(pixels[:,1]>150)&(pixels[:,2]<120)).mean()
            white=(pixels.min(axis=1)>215).mean()
            kind = "electric" if not flame and spark>.08 and white>.04 else "normal"
            info.append((index, direction, quality, tuple(int(v) for v in np.rint(click)), color, kind,
                         tuple(map(int,(x+bx,y+by,bw,bh))) if flame else None, None))
        except RecognitionError as exc:
            bx, by, bw, bh = bbox
            special = find_timer_shape(image, (x+bx, y+by, bw, bh), line_width)
            if special:
                path, direction, tip, visual_rect, head_rect = special
                paths.append(path)
                info.append((index, direction, .85, tip, (170,170,170), "timer", visual_rect, head_rect))
                timer_heads.append(head_rect)
            else:
                failures.append((index, (x+bx,y+by,bw,bh), str(exc)))
    for index, (fx,fy,fw,fh), error in failures:
        # Detached eye/mouth marks belong to a positively identified sprite only.
        if any(hx <= fx and hy <= fy and fx+fw <= hx+hw and fy+fh <= hy+hh
               for hx,hy,hw,hh in timer_heads):
            continue
        issues.append(f"区域 {index}（原图 x={fx}, y={fy}, {fw}×{fh}）：{error}")
    if not paths:
        raise RecognitionError("未识别到可靠箭头。" + "；".join(issues[:4]))
    try:
        origin, step = fit_grid(paths, line_width)
    except RecognitionError:
        if not channels:
            raise
        # Sparse lane tutorials have no horizontal arrow shafts. Coordinates
        # remain a display lattice; all lane-board collisions use pixel geometry.
        spacing = min(min(c.visual_rect[2:])*.5 for c in channels)
        origin = tuple(np.concatenate(paths).min(axis=0))
        step = (spacing,spacing)
    combined = np.concatenate(paths+[np.array(c.pixel_path) for c in channels])
    minimum = np.rint((combined.min(axis=0) - origin) / step).astype(int)
    origin = tuple(float(origin[i] + minimum[i] * step[i]) for i in range(2))
    max_cell = np.rint((combined.max(axis=0) - origin) / step).astype(int)
    board = Board(w, h, (x, y, cw, ch), origin, tuple(step), int(max_cell[0]) + 1, int(max_cell[1]) + 1,
                  [], image_hash(image), issues, line_width=line_width)
    for path, (aid, direction, quality, click, color, kind, visual_rect, head_rect) in zip(paths, info):
        cells = []
        vertices = cv2.approxPolyDP(path.astype(np.float32).reshape(-1,1,2),line_width*.5,False)[:,0]
        deltas = np.diff(vertices,axis=0)
        skew = any(min(abs(d)) > line_width*1.5 and abs(abs(d[0])-abs(d[1])) >= line_width for d in deltas)
        diagonal = any(min(abs(d)) > line_width*1.5 and abs(abs(d[0])-abs(d[1])) < line_width for d in deltas)
        if diagonal and not skew:
            # Quantize corners, then fill true 45-degree links instead of pixel stair-steps.
            corners = [board.cell(tuple(p)) for p in vertices]
            expanded = [corners[0]]
            for target in corners[1:]:
                start = expanded[-1]
                dx,dy = target[0]-start[0],target[1]-start[1]
                if dx and dy and abs(dx)!=abs(dy):
                    issues.append(f"箭头 {aid} 的斜线未对齐 45 度网格，请核对")
                    break
                steps=max(abs(dx),abs(dy))
                if steps:
                    expanded.extend((start[0]+i*(dx//steps),start[1]+i*(dy//steps)) for i in range(1,steps+1))
            path = np.array([board.pixel(c) for c in expanded])
        for point in path:
            cell = board.cell(tuple(point))
            if not cells or cells[-1] != cell:
                cells.append(cell)
        arrow = Arrow(aid, cells, direction, click, quality, color, kind, visual_rect, head_rect)
        arrow.pixel_path = [tuple(map(float,p)) for p in vertices]
        if skew:
            vector = np.array(DIRECTIONS[direction],dtype=float)
            arrow.launch_vectors = [tuple(vector/np.linalg.norm(vector))]
            terminal = deltas[-1].astype(float)
            # Until gameplay confirms its angle, require both the discrete head
            # direction and the actual oblique shaft continuation to be clear.
            if min(abs(terminal)) > line_width*1.5 and abs(abs(terminal[0])-abs(terminal[1])) >= line_width:
                tangent = terminal/np.linalg.norm(terminal)
                if np.dot(tangent,arrow.launch_vectors[0]) < .94:
                    issues.append(f"箭头 {aid} 的真实末段与头部方向冲突，请核对")
                else:
                    arrow.launch_vectors.append(tuple(tangent))
        if kind == "timer":
            # Rounding the sprite nose can reserve the neighboring arrow's tail
            # cell, although that cell lies ahead of the actual physical nose.
            # Keep the last body cell behind the nose; the ray then checks the
            # adjacent cell instead of silently including it in the missile.
            vector = np.array(DIRECTIONS[direction])
            occupied = list(arrow.cells)
            while len(occupied) > 1 and np.dot(np.array(board.pixel(occupied[-1]))-click, vector) > line_width*.25:
                occupied.pop()
            if occupied != arrow.cells:
                arrow.footprint = occupied
        if kind == "wide":
            (_, _, _, filled), offset = wide_shapes[aid]
            # A straight broad arrow reserves its maximum width along the shaft.
            # Work in grid units, avoiding one-pixel changes in anti-aliased edges.
            yy,xx = np.nonzero(filled)
            horizontal = direction in ("LEFT","RIGHT")
            axis = 0 if horizontal else 1
            length_values = (xx if horizontal else yy) + offset[axis]
            grid_values = (length_values-origin[axis])/step[axis]
            lo,hi = int(np.ceil(grid_values.min()-.05)), int(np.floor(grid_values.max()+.05))
            cross_values = (yy if horizontal else xx)+offset[1-axis]
            radius = int(np.floor((np.ptp(cross_values)+1)/(2*step[1-axis])+.05))
            center = cells[0][1-axis]
            # Nose outside a grid center must not overlap the next ordinary tail.
            arrow.cells = [cell for cell in cells if lo <= cell[axis] <= hi]
            arrow.footprint = sorted((along,cross) if horizontal else (cross,along)
                                     for along in range(lo,hi+1)
                                     for cross in range(center-radius,center+radius+1))
        board.arrows.append(arrow)
    # The conservative maximum-width footprint can extend past a decorated
    # tail cap. A visible neighboring arrow in those empty pixels remains an
    # independent instance; trim only unsupported cells on the tail row.
    from .geometry import point_distance
    for broad in board.arrows:
        if broad.kind!='wide':
            continue
        (_,_,_,filled),offset = wide_shapes[broad.id]
        axis = 0 if broad.direction in ('LEFT','RIGHT') else 1
        tail_row = broad.cells[0][axis]
        other_cells = {c for other in board.arrows if other.id!=broad.id and other.kind=='normal'
                       for c in other.cells if any(point_distance(board.pixel(c),a,b)<=line_width*.5
                                                   for a,b in zip(other.pixel_path,other.pixel_path[1:]))}
        trimmed = []
        for cell in broad.footprint:
            px,py = np.rint(np.array(board.pixel(cell))-offset).astype(int)
            supported = 0<=py<filled.shape[0] and 0<=px<filled.shape[1] and filled[py,px]>0
            if cell[axis]!=tail_row or cell not in other_cells or supported:
                trimmed.append(cell)
        broad.footprint = trimmed
    from .models import Portal, CounterBlock
    board.portals = [Portal(board.cell(center), center, rect) for center,rect in portals]
    board.counter_blocks = [CounterBlock(board.cell(center), value, rect) for center,rect,value in counter_blocks]
    board.turn_channels = channels
    board.orbit_obstacle = orbit
    for channel in channels:
        if channel.remaining is None:
            board.issues.append(f'转弯通道 {channel.id} 的剩余次数不确定')
    for block in board.counter_blocks:
        if block.remaining is None:
            board.issues.append(f"数字方块 {block.cell} 的数字不确定，请在棋盘 JSON 中校正")
    if timer_heads:
        board.timer_seconds, board.timer_reading = read_countdown(image)
    return Recognition(board, foreground)
