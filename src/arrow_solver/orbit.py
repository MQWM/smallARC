"""Rectangular moving obstacle detection and a simple exit timing filter."""
from math import isfinite
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
    # Sandglass green caps lie on the same rail. Ignore only compact rail objects.
    green = cv2.inRange(hsv,(35,100,150),(90,255,255))
    _,_,stats,centers = cv2.connectedComponentsWithStats(green)
    for s,p in zip(stats[1:],centers[1:]):
        if s[4]>radius*radius*.08 and max(s[2:4])<radius*3 and on_edge(p):
            r = radius*1.2
            ignored.append((round(p[0]-r),round(p[1]-r),round(r*2),round(r*2)))
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


def exit_phases(board,arrow,counts):
    from .geometry import launch_segments,rectangle_segments
    if board.turn_channels:
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


def wait_seconds(obstacle,exits,elapsed,period=37.,window=3.,clockwise=True,line_width=8.):
    if not all(isfinite(v) for v in (elapsed,period,window)) or period<=0 or window<0 or elapsed<0:
        raise ValueError('小豆豆速度、避让窗口或抓图时间无效')
    perimeter = 2*sum(obstacle.track_rect[2:])
    speed = perimeter/period
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
