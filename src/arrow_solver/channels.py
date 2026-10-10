"""Finite-use turning lanes: detect ports and simulate routes for each board state."""
from math import hypot
import time
import cv2
import numpy as np
from .models import TurnChannel,DIRECTIONS,Plan
from . import geometry


def _channel_path(ports, rect, badge_size):
    """Build L/U centerlines from two ports, independent of skin rotation."""
    rank = {'top':0,'right':1,'bottom':2,'left':3}
    ports = sorted(ports,key=lambda p:(rank[p[0]],p[1][0],p[1][1]))
    (side,entry),(out_side,exit) = ports
    x,y,w,h = rect
    if side==out_side:
        inset = badge_size*.25
        if side in ('top','bottom'):
            bend = y+h-inset if side=='top' else y+inset
            path = [entry,(entry[0],bend),(exit[0],bend),exit]
        else:
            bend = x+w-inset if side=='left' else x+inset
            path = [entry,(bend,entry[1]),(bend,exit[1]),exit]
    elif (rank[side]-rank[out_side])%2:
        corner = (entry[0],exit[1]) if side in ('top','bottom') else (exit[0],entry[1])
        path = [entry,corner,exit]
    else:
        # Opposite-side ports are not one of the confirmed L/U styles.
        return None
    inward = {'top':'DOWN','right':'LEFT','bottom':'UP','left':'RIGHT'}
    outward = {'top':'UP','right':'RIGHT','bottom':'DOWN','left':'LEFT'}
    return entry,exit,inward[side],outward[out_side],path


def find_channels(image,crop):
    from .special import counter_digit_templates,normalize_digit
    x,y,w,h = crop
    hsv = cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv,(103,35,100),(117,175,255))
    _,_,stats,_ = cv2.connectedComponentsWithStats(blue)
    found = []
    for bx,by,bw,bh,area in stats[1:]:
        if min(bw,bh)<25 or not .75<bw/bh<1.4 or area/(bw*bh)<.4:
            continue
        patch = hsv[by:by+bh,bx:bx+bw]
        white = cv2.inRange(patch,(0,0,220),(180,45,255))
        _,_,badges,_ = cv2.connectedComponentsWithStats(white)
        badges = [r for r in badges[1:] if min(r[2:4])>min(bw,bh)*.25 and .75<r[2]/r[3]<1.3 and r[4]/(r[2]*r[3])>.4]
        if len(badges)!=1:
            continue
        qx,qy,qw,qh,_ = badges[0]
        yy,xx = np.mgrid[:bh,:bw]
        inner = ((xx-(qx+(qw-1)/2))**2+(yy-(qy+(qh-1)/2))**2)<(min(qw,qh)*.44)**2
        ink = inner & (patch[:,:,1]>30) & (patch[:,:,2]<220)
        value = None
        if ink.any():
            target = normalize_digit(ink)
            score,digit = max((float((target&t).sum()/max(1,(target|t).sum())),d) for d,t in counter_digit_templates())
            if score>=.48 and digit>0:
                value = digit
        rim = cv2.inRange(patch,(103,25,100),(117,175,195))
        k = max(3,round(min(bw,bh)*.12))
        ports = []
        for side,region in [('top',rim[:k]),('bottom',rim[-k:]),('left',rim[:,:k]),('right',rim[:,-k:])]:
            _,_,parts,centers = cv2.connectedComponentsWithStats(region)
            for (px,py,pw,ph,pa),center in zip(parts[1:],centers[1:]):
                span = pw if side in ('top','bottom') else ph
                if span<min(bw,bh)*.35 or pa/(pw*ph)<.6:
                    continue
                point = (x+bx+float(center[0]),y+by) if side=='top' else (x+bx+float(center[0]),y+by+bh-1) if side=='bottom' else (x+bx,y+by+float(center[1])) if side=='left' else (x+bx+bw-1,y+by+float(center[1]))
                ports.append((side,point))
        if len(ports)!=2:
            # Retain the unexplained component so recognition refuses execution.
            continue
        visual_rect = (int(x+bx),int(y+by),int(bw),int(bh))
        result = _channel_path(ports,visual_rect,min(qw,qh))
        if result is None:
            continue
        entry,exit,entry_direction,exit_direction,route = result
        entry,exit = tuple(map(float,entry)),tuple(map(float,exit))
        route = [tuple(map(float,p)) for p in route]
        found.append(TurnChannel(len(found)+1,value,entry,exit,entry_direction,
                                 exit_direction,route,visual_rect))
    return found


def _hit(start,direction,rect):
    x,y,w,h = rect
    low,high = 0.,float('inf')
    for s,d,a,b in zip(start,direction,(x,y),(x+w,y+h)):
        if abs(d)<1e-9:
            if not a<=s<=b:
                return None
        else:
            t,u = sorted(((a-s)/d,(b-s)/d))
            low,high = max(low,t),min(high,u)
    return low if high>1e-6 and low<=high else None


def route(board,arrow,counts):
    start = arrow.pixel_path[-1]
    direction = DIRECTIONS[arrow.direction]
    segments,uses = [],[]
    last = None
    length = hypot(board.image_width,board.image_height)*2
    radius = board.line_width*.5
    while True:
        norm = hypot(*direction)
        d = (direction[0]/norm,direction[1]/norm)
        stop = length
        for portal in board.portals:
            delta = (portal.center[0]-start[0],portal.center[1]-start[1])
            along = delta[0]*d[0]+delta[1]*d[1]
            if along>0 and abs(delta[0]*d[1]-delta[1]*d[0])<=radius:
                stop = min(stop,along)
        hits = []
        for channel in board.turn_channels:
            if counts[channel.id]<=0 or channel.id==last:
                continue
            t = _hit(start,d,channel.visual_rect)
            if t is not None and t<stop:
                hits.append((t,channel.id,channel))
        if not hits:
            segments.append((start,(start[0]+d[0]*stop,start[1]+d[1]*stop)))
            return segments,uses,False
        _,_,channel = min(hits,key=lambda h:h[:2])
        candidates = []
        orientations = [(DIRECTIONS[channel.entry_direction],channel.pixel_path,
                         DIRECTIONS[channel.exit_direction]),
                        (tuple(-v for v in DIRECTIONS[channel.exit_direction]),
                         list(reversed(channel.pixel_path)),
                         tuple(-v for v in DIRECTIONS[channel.entry_direction]))]
        for incoming,path,outgoing in orientations:
            delta = (path[0][0]-start[0],path[0][1]-start[1])
            along = delta[0]*d[0]+delta[1]*d[1]
            aligned = abs(delta[0]*d[1]-delta[1]*d[0])<=board.line_width*.75
            if direction==incoming and aligned and along>=0:
                candidates.append((along,path,outgoing))
        if channel.id in uses or not candidates:
            return segments,uses,True
        _,path,outgoing = min(candidates,key=lambda item:item[0])
        segments.append((start,path[0]))
        segments.extend(zip(path,path[1:]))
        uses.append(channel.id)
        start,direction,last = path[-1],outgoing,channel.id


def move(board,arrow,remaining,counts):
    segments,uses,blocked = route(board,arrow,counts)
    blockers = []
    for other in board.arrows:
        if other.id==arrow.id or other.id not in remaining:
            continue
        bodies,radius = geometry.body_segments(board,other)
        if any(geometry.segment_distance(a,b,c,d)<=radius+board.line_width*.5
               for a,b in segments for c,d in bodies):
            blockers.append(other.id)
    threshold = 0
    for block in board.counter_blocks:
        x,y,w,h = block.visual_rect
        if any(x<=a[0]<=x+w and y<=a[1]<=y+h or
               any(geometry.segment_distance(a,b,c,d)<=board.line_width*.5 for c,d in geometry.rectangle_segments(block.visual_rect))
               for a,b in segments):
            threshold = max(threshold,block.remaining)
    return blocked,blockers,uses,threshold


def validate(board):
    from .solver import BoardError
    if not board.arrows or len({a.id for a in board.arrows})!=len(board.arrows):
        raise BoardError('箭头编号或数量无效')
    for a in board.arrows:
        if a.kind!='normal' or a.launch_vectors:
            raise BoardError('转弯通道组合目前仅支持普通箭头，请人工处理特殊箭头')
        if a.direction not in DIRECTIONS or not a.cells:
            raise BoardError('箭头路径或方向无效')
        geometry.validate_geometry(board,a)
    if len({c.id for c in board.turn_channels})!=len(board.turn_channels):
        raise BoardError('转弯通道编号重复')
    for c in board.turn_channels:
        if c.remaining is None or c.remaining<1 or c.entry_direction not in DIRECTIONS or c.exit_direction not in DIRECTIONS or len(c.pixel_path)<2:
            raise BoardError('转弯通道方向、次数或路径不确定')


def solve(board):
    from .solver import BoardError
    validate(board)
    counts = {c.id:c.remaining for c in board.turn_channels}
    ids = frozenset(a.id for a in board.arrows)
    arrows = {a.id:a for a in board.arrows}
    routes = {}
    def movement(aid,state):
        # Positive counts share the same geometry. Cache body intersections
        # once per active-channel layout, then filter removed arrows per state.
        key = (aid,tuple(sorted((cid,n>0) for cid,n in state.items())))
        if key not in routes:
            routes[key] = move(board,arrows[aid],ids,state)
        return routes[key]
    initial = {aid:movement(aid,counts)[1] for aid in ids}
    failed = set()
    nodes = 0
    deadline = time.monotonic()+15
    def search(remaining,state):
        nonlocal nodes
        nodes += 1
        if nodes>50000 or time.monotonic()>deadline:
            raise BoardError('转弯通道规划搜索过大，已停止')
        if not remaining:
            return []
        key = (remaining,tuple(sorted(state.items())))
        if key in failed:
            return None
        successful = len(ids)-len(remaining)
        choices = []
        for aid in sorted(remaining):
            blocked,blockers,uses,threshold = movement(aid,state)
            if not blocked and not (set(blockers)&remaining) and threshold<=successful:
                choices.append((threshold,aid,uses))
        for threshold,aid,uses in sorted(choices):
            after = state.copy()
            for cid in uses:
                after[cid]-=1
            rest = search(remaining-{aid},after)
            if rest is not None:
                return [(aid,uses,threshold)]+rest
        failed.add(key)
        return None
    steps = search(ids,counts)
    if steps is None:
        raise BoardError('转弯通道动态路径没有完整可执行顺序，请检查入口、出口、数字及箭头')
    return Plan([a for a,_,_ in steps],initial,board.fingerprint(),
                {a:uses for a,uses,_ in steps},{a:t for a,_,t in steps})


def verify(board,plan):
    from .solver import BoardError
    validate(board)
    remaining = {a.id for a in board.arrows}
    counts = {c.id:c.remaining for c in board.turn_channels}
    if plan.board_fingerprint!=board.fingerprint() or len(plan.order)!=len(remaining) or set(plan.order)!=remaining:
        raise BoardError('棋盘或转弯通道计划失效')
    arrows = {a.id:a for a in board.arrows}
    for index,aid in enumerate(plan.order):
        blocked,blockers,uses,threshold = move(board,arrows[aid],remaining,counts)
        if blocked or blockers or threshold>index:
            raise BoardError(f'箭头 {aid} 的动态转弯路径仍被阻挡')
        if plan.channel_uses.get(aid)!=uses or plan.counter_thresholds.get(aid)!=threshold:
            raise BoardError(f'箭头 {aid} 的转弯通道计划状态不一致')
        for cid in uses:
            counts[cid]-=1
        remaining.remove(aid)
