"""Rectangular moving obstacle detection and a simple exit timing filter."""
from math import isfinite
from dataclasses import dataclass
import time
import cv2
import numpy as np
from .models import OrbitObstacle


def detect(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    height,width = image.shape[:2]
    dots = cv2.inRange(hsv,(10,120,140),(30,255,255))
    _,_,stats,centers = cv2.connectedComponentsWithStats(dots)
    points = np.array([p for s,p in zip(stats[1:],centers[1:])
                       if 2<=s[4]<=max(40,width*width*.00005) and max(s[2:4])<=width*.012
                       and height*.2<p[1]<height*.9])
    if len(points)<32:
        return None
    left,top = points.min(axis=0)
    right,bottom = points.max(axis=0)
    tolerance = max(3,width*.004)
    if right-left<width*.4 or bottom-top<height*.2:
        return None
    if any(np.count_nonzero(abs(points[:,axis]-value)<tolerance)<8
           for axis,value in [(0,left),(0,right),(1,top),(1,bottom)]):
        return None
    def on_edge(center):
        x,y = center
        return left-tolerance<=x<=right+tolerance and top-tolerance<=y<=bottom+tolerance and min(abs(x-left),abs(x-right),abs(y-top),abs(y-bottom))<width*.025
    yellow = cv2.inRange(hsv,(18,160,200),(40,255,255))
    _,labels,stats,centers = cv2.connectedComponentsWithStats(yellow)
    found = []
    for i,(s,center) in enumerate(zip(stats[1:],centers[1:]),1):
        x,y,w,h,area = s
        if area<width*width*.0004 or not .65<w/h<1.5 or max(w,h)>width*.1 or not on_edge(center):
            continue
        contours,_ = cv2.findContours((labels==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        center,radius = cv2.minEnclosingCircle(max(contours,key=cv2.contourArea))
        if on_edge(center):
            found.append((center,float(radius),(int(x)-2,int(y)-2,int(w)+4,int(h)+4)))
    if len(found)!=1:
        return None
    center,radius,rect = found[0]
    ignored = [rect]
    # Ignore compact objects straddling the rail, without classifying their skin.
    # Keep this spatial test narrow: nearby arrows inside the board must survive.
    from .vision import foreground_mask
    _,_,stats,_ = cv2.connectedComponentsWithStats(foreground_mask(image))
    for bx,by,bw,bh,area in stats[1:]:
        vertical = (bx-tolerance<=left<=bx+bw+tolerance or bx-tolerance<=right<=bx+bw+tolerance) and by<=bottom and by+bh>=top
        horizontal = (by-tolerance<=top<=by+bh+tolerance or by-tolerance<=bottom<=by+bh+tolerance) and bx<=right and bx+bw>=left
        if area>radius*radius*.08 and max(bw,bh)<radius*3 and (vertical or horizontal):
            ignored.append((int(bx)-2,int(by)-2,int(bw)+4,int(bh)+4))
    return OrbitObstacle(tuple(map(float,(left,top,right-left,bottom-top))),
                         tuple(map(float,center)),radius,ignored)


def mask_objects(mask,crop,obstacle):
    if obstacle is None:
        return mask
    x,y,w,h = crop
    for bx,by,bw,bh in obstacle.ignored_regions:
        left,right = max(0,bx-x),min(w,bx+bw-x)
        top,bottom = max(0,by-y),min(h,by+bh-y)
        if left<right and top<bottom:
            mask[top:bottom,left:right] = 0
    return mask


def phase(rect,point):
    x,y,w,h = rect
    px,py = point
    # Clockwise: top -> right -> bottom -> left.
    side = min(range(4),key=lambda i:[abs(py-y),abs(px-x-w),abs(py-y-h),abs(px-x)][i])
    return [max(0,min(w,px-x)),w+max(0,min(h,py-y)),
            w+h+max(0,min(w,x+w-px)),2*w+h+max(0,min(h,y+h-py))][side]


def position(rect, distance):
    """Inverse of clockwise phase, including corner and lap crossings."""
    x,y,w,h = rect
    distance %= 2*(w+h)
    if distance <= w:
        return x+distance,y
    distance -= w
    if distance <= h:
        return x+w,y+distance
    distance -= h
    if distance <= w:
        return x+w-distance,y+h
    return x,y+h-(distance-w)


def predicted_position(obstacle, elapsed, speed=118., clockwise=True):
    sign = 1 if clockwise else -1
    return position(obstacle.track_rect,phase(obstacle.track_rect,obstacle.center)+sign*speed*max(0.,elapsed))


def measure(first, second, interval):
    if not isfinite(interval) or not .25 <= interval <= 10:
        raise ValueError('两帧间隔不足或过长，无法测量小豆豆速度')
    tolerance = max(6., second.track_rect[2]*.01)
    if max(abs(a-b) for a,b in zip(first.track_rect,second.track_rect)) > tolerance:
        raise ValueError('两帧轨道范围不同，请重新抓图')
    perimeter = 2*sum(second.track_rect[2:])
    # Use rail distance, not straight-line displacement, when crossing corners.
    distance = (phase(second.track_rect,second.center)-phase(second.track_rect,first.center)+perimeter/2)%perimeter-perimeter/2
    speed = abs(distance)/interval
    if abs(distance) < max(2.,second.radius*.1) or abs(distance) >= perimeter/4 or not 1 <= speed <= 3000:
        raise ValueError('小豆豆位移不足或有歧义，无法确认移动速度')
    second.measured_speed = speed
    second.measured_clockwise = distance > 0
    second.sample_interval = interval
    return second


@dataclass
class OrbitCapture:
    image: np.ndarray
    captured_at: float
    obstacle: OrbitObstacle | None
    warning: str = ''


def capture_pair(device, control, *, first_image=None, first_at=None,
                 now=time.monotonic, interval=.8, report=lambda kind,data:None):
    """At most two pre-execution frames; the second frame becomes the source."""
    def shot():
        control.check()
        start = now()
        image = device.screenshot()
        end = now()
        control.check()
        return image,(start+end)/2
    if first_image is None:
        first_image,first_at = shot()
    if first_at is None:
        raise ValueError('缺少第一帧抓图时间')
    first = detect(first_image)
    if first is None:
        return OrbitCapture(first_image,first_at,None)
    report('orbit_sampling',{'message':'检测到小豆豆，正在获取第二帧测量速度…'})
    control.wait(max(0.,first_at+interval-now()))
    image,captured_at = shot()
    second = detect(image)
    try:
        if image.shape != first_image.shape or second is None:
            raise ValueError('第二帧未确认同一轨道和小豆豆，请重新抓图')
        measure(first,second,captured_at-first_at)
    except ValueError as exc:
        # Keep a detected position, but never disguise fallback as measurement.
        return OrbitCapture(image if second else first_image,
                            captured_at if second else first_at,second or first,str(exc))
    return OrbitCapture(image,captured_at,second)


def exit_phases(board,arrow,counts):
    from .geometry import launch_segments,rectangle_segments
    if board.turn_channels or board.teleport_gates:
        from .channels import route
        segments,_,blocked = route(board,arrow,counts)
        if blocked:
            raise ValueError('转弯通道出口被封闭侧面阻挡')
    else:
        segments = [(a,b) for a,b,_ in launch_segments(board,arrow)]
    results = []
    for a,b in segments:
        dx,dy = b[0]-a[0],b[1]-a[1]
        for c,d in rectangle_segments(board.orbit_obstacle.track_rect):
            ux,uy = d[0]-c[0],d[1]-c[1]
            cross = dx*uy-dy*ux
            if abs(cross)<1e-9:
                continue
            wx,wy = c[0]-a[0],c[1]-a[1]
            t = (wx*uy-wy*ux)/cross
            u = (wx*dy-wy*dx)/cross
            if 0<=t<=1 and 0<=u<=1:
                results.append(phase(board.orbit_obstacle.track_rect,(a[0]+t*dx,a[1]+t*dy)))
    return results


def wait_seconds(obstacle,exits,elapsed,speed=118.,window=3.,clockwise=True,line_width=8.):
    if not all(isfinite(v) for v in (elapsed,speed,window)) or speed<=0 or window<0 or elapsed<0:
        raise ValueError('小豆豆速度、避让窗口或抓图时间无效')
    perimeter = 2*sum(obstacle.track_rect[2:])
    period = perimeter/speed
    sign = 1 if clockwise else -1
    current = (phase(obstacle.track_rect,obstacle.center)+sign*speed*elapsed)%perimeter
    # Fixed capture/dispatch margin plus a small allowance for guessed speed drift.
    pad = (obstacle.radius+line_width*.5)/speed+.5+elapsed*.005
    if window+2*pad>=period:
        raise ValueError('小豆豆位置预测已过期，请重新抓图')
    waits = []
    for target in exits:
        arrival = (sign*(target-current)%perimeter)/speed
        if arrival<=window+pad:
            waits.append(arrival+pad)
        elif period-arrival<=pad:
            waits.append(pad-(period-arrival))
    return max(waits,default=0.)
