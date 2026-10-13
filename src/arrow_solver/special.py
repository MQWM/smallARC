"""Shape-based recovery of the outlined countdown bullet; no timing rules inferred from color."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np

from .models import DIRECTIONS


def flame_arrow_mask(image, rect, mask):
    """Normalize a red/yellow flame skin; its gameplay remains ordinary."""
    x,y,w,h=rect
    hsv=cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    pixels=hsv[mask[3:-3,3:-3]>0]
    red=(pixels[:,0]<12)&(pixels[:,1]>180)&(pixels[:,2]>160)
    yellow=(pixels[:,0]>=12)&(pixels[:,0]<40)&(pixels[:,1]>150)&(pixels[:,2]>180)
    if not len(pixels) or red.mean()<.15 or yellow.mean()<.15:
        return None
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(mask)
    cv2.drawContours(filled,contours,-1,255,-1)
    distance=cv2.distanceTransform(filled,cv2.DIST_L2,5)
    medial=(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8)))&(distance>1)
    if not medial.any():
        return None
    return filled,2*float(np.median(distance[medial]))


def timer_mask(image, visual_rect, head_rect):
    result = np.zeros(image.shape[:2], np.uint8)
    x, y, w, h = map(int, visual_rect)
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Neutral body + dark outline; the blue background is saturated.
    result[y:y+h, x:x+w] = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 20)).astype(np.uint8) * 255
    hx, hy, hw, hh = map(int, head_rect)
    result[hy:hy+hh, hx:hx+hw] = 255
    return result


def arrow_components(image, foreground, line_width=None):
    """Shared recognition/preflight partition for confirmed touching arrows."""
    count,labels,stats,centers = timer_separated_components(image,foreground,line_width)
    if line_width is None:
        distance = cv2.distanceTransform(foreground,cv2.DIST_L2,5)
        medial = (distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))) & (distance>1)
        line_width = 2*float(np.median(distance[medial])) if medial.any() else 2.
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    warm = (hsv[:,:,0]<40)&(hsv[:,:,1]>100)&(hsv[:,:,2]>130)
    minimum = max(12,image.shape[0]*image.shape[1]*.000015)
    for label in range(1,count):
        x,y,w,h,area = stats[label]
        if min(w,h)<line_width*2.5 or area<line_width**2*8:
            continue
        patch = labels[y:y+h,x:x+w]
        original = patch==label
        skin = original & warm[y:y+h,x:x+w]
        shape = find_wide_shape(np.pad(skin.astype(np.uint8)*255,3),line_width)
        if shape is None:
            continue
        filled = shape[3][3:-3,3:-3]>0
        yy,xx = np.nonzero(filled)
        envelope = np.zeros_like(filled)
        envelope[yy.min():yy.max()+1,xx.min():xx.max()+1] = True
        # Faces may protrude through small gaps in the colored shaft contour.
        # Neutral decoration inside its bounds is part of the skin, not a head.
        claimed = filled | (envelope & (hsv[y:y+h,x:x+w,1]<60))
        outside = original & ~claimed
        # A colored neighbor must have a visible, independently recognizable head.
        n,parts,part_stats,_ = cv2.connectedComponentsWithStats(outside.astype(np.uint8))
        neighbors = []
        from .vision import find_head,RecognitionError
        for part in range(1,n):
            px,py,pw,ph,pa = part_stats[part]
            if pa<minimum:
                continue
            mask = np.pad((parts[py:py+ph,px:px+pw]==part).astype(np.uint8)*255,3)
            try:
                find_head(mask,line_width)
            except RecognitionError:
                continue
            neighbors.append(part)
        if not neighbors:
            continue
        # Claim internal decoration for the skin, including separate eye fragments.
        patch[claimed & (foreground[y:y+h,x:x+w]>0)] = label
        for part in neighbors:
            patch[parts==part] = count
            count+=1
    if count!=len(stats):
        # Rebuild bounding boxes for split components and the broad skin.
        new_stats = np.zeros((count,5),np.int32)
        new_centers = np.zeros((count,2),float)
        for label in range(count):
            yy,xx = np.nonzero(labels==label)
            if len(xx):
                new_stats[label] = (xx.min(),yy.min(),np.ptp(xx)+1,np.ptp(yy)+1,len(xx))
                new_centers[label] = (xx.mean(),yy.mean())
        stats,centers = new_stats,new_centers
    return color_separated_components(image,count,labels,stats,centers)


def color_separated_components(image,count,labels,stats,centers):
    """Split two distinct solid skins only when both recover complete arrows."""
    from .vision import find_head,trace,RecognitionError
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    changed = False
    for label in range(1,len(stats)):
        x,y,w,h,area = stats[label]
        if area<40 or min(w,h)<5:
            continue
        original = labels[y:y+h,x:x+w]==label
        pixels = hsv[y:y+h,x:x+w]
        colored = (pixels[:,:,1]>55)&(pixels[:,:,2]>100)&original
        if colored.sum()<area*.95:
            continue
        hist = np.bincount(pixels[:,:,0][colored],minlength=180)
        smooth = sum(np.roll(hist,i) for i in range(-3,4))
        first = int(smooth.argmax())
        hues = np.arange(180)
        distance = np.minimum(abs(hues-first),180-abs(hues-first))
        second_scores = np.where(distance>=18,smooth,0)
        second = int(second_scores.argmax())
        minimum = max(20,area*.08)
        if second_scores[second]<minimum:
            continue
        # More than two substantial colors may be a special skin or ambiguous
        # multi-arrow contact. Leave it for existing recognition to explain.
        d2 = np.minimum(abs(hues-second),180-abs(hues-second))
        if np.any(smooth[(distance>=18)&(d2>=18)]>=minimum):
            continue
        mask = np.pad(original.astype(np.uint8)*255,3)
        dt = cv2.distanceTransform(mask,cv2.DIST_L2,5)
        medial = (dt>=cv2.dilate(dt,np.ones((3,3),np.uint8)))&(dt>1)
        width = 2*float(np.median(dt[medial])) if medial.any() else 2.
        try:
            tip,direction,_ = find_head(mask,width)
            trace(mask,tip,direction,width)
            continue  # A valid whole arrow, including multicolor skins, stays intact.
        except RecognitionError:
            pass
        hue = pixels[:,:,0].astype(np.int16)
        dist1 = np.minimum(abs(hue-first),180-abs(hue-first))
        dist2 = np.minimum(abs(hue-second),180-abs(hue-second))
        parts = [original&(dist1<=dist2),original&(dist2<dist1)]
        if any(part.sum()<minimum for part in parts):
            continue
        valid = True
        for part in parts:
            # No discarded pixels, invented tails, or weakened branch checks.
            yy,xx = np.nonzero(part)
            part_mask = np.pad(part[yy.min():yy.max()+1,xx.min():xx.max()+1].astype(np.uint8)*255,3)
            try:
                tip,direction,_ = find_head(part_mask,width)
                trace(part_mask,tip,direction,width)
            except RecognitionError:
                valid = False
                break
        if valid:
            labels[y:y+h,x:x+w][parts[1]] = count
            count += 1
            changed = True
    if changed:
        stats = np.zeros((count,5),np.int32)
        centers = np.zeros((count,2),float)
        for label in range(count):
            yy,xx = np.nonzero(labels==label)
            if len(xx):
                stats[label] = (xx.min(),yy.min(),np.ptp(xx)+1,np.ptp(yy)+1,len(xx))
                centers[label] = (xx.mean(),yy.mean())
    return count,labels,stats,centers


def timer_separated_components(image, foreground, line_width=None):
    """Give colored neighbors of a confirmed missile their own foreground labels.

    Color alone never identifies a missile. Its neutral shaft and outlined head
    must first pass shape recovery; colored marks inside the head stay decoration.
    Both recognition and execution use this partition so missing neighbors remain
    unexplained foreground during execution preflight.
    """
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    neutral = (hsv[:,:,1] < 35) & (hsv[:,:,2] > 110) & (hsv[:,:,2] < 205)
    gray_counts = np.bincount(labels[neutral], minlength=count)
    white = (hsv[:,:,1] < 50) & (hsv[:,:,2] > 185)
    white_counts = np.bincount(labels[white], minlength=count)
    if not np.any((gray_counts[1:] >= 20) & (white_counts[1:] >= 4)):
        return count, labels, stats, centers
    if line_width is None:
        distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        medial = (distance >= cv2.dilate(distance, np.ones((3,3),np.uint8))) & (distance > 1)
        line_width = 2*float(np.median(distance[medial])) if medial.any() else 2.
    minimum_area = max(12, image.shape[0]*image.shape[1]*.000015)
    added = []
    added_centers = []
    for label in range(1, count):
        if gray_counts[label] < line_width**2*5 or white_counts[label] < line_width**2:
            continue
        x,y,w,h,_ = stats[label]
        special = find_timer_shape(image, (int(x),int(y),int(w),int(h)), line_width)
        if special is None:
            continue
        hx,hy,hw,hh = special[4]
        colored = ((labels[y:y+h,x:x+w] == label) &
                   (hsv[y:y+h,x:x+w,1] > 70) & (hsv[y:y+h,x:x+w,2] > 150))
        # The face may contain colored eyes/mouth; do not extract them as arrows.
        colored[max(0,hy-y):min(h,hy+hh-y),max(0,hx-x):min(w,hx+hw-x)] = False
        n, parts, part_stats, _ = cv2.connectedComponentsWithStats(colored.astype(np.uint8))
        split = False
        for part in range(1,n):
            px,py,pw,ph,area = part_stats[part]
            if area < minimum_area:
                continue
            new_label = count + len(added)
            labels[y:y+h,x:x+w][parts == part] = new_label
            added.append((x+px,y+py,pw,ph,area))
            yy,xx = np.nonzero(parts == part)
            added_centers.append((x+xx.mean(),y+yy.mean()))
            split = True
        if split:
            yy,xx = np.nonzero(labels[y:y+h,x:x+w] == label)
            stats[label] = (x+xx.min(),y+yy.min(),np.ptp(xx)+1,np.ptp(yy)+1,len(xx))
            centers[label] = (x+xx.mean(),y+yy.mean())
    if added:
        stats = np.vstack((stats,np.asarray(added,dtype=stats.dtype)))
        centers = np.vstack((centers,added_centers))
        count += len(added)
    return count, labels, stats, centers


def find_timer_shape(image, rect, line_width):
    # Prefer the original crisp outline. Only widen grouping when antialiasing
    # has fragmented it; geometry and shaft tracing must still both succeed.
    for threshold, radius_scale in ((215,.25),(185,.5),(185,.75)):
        shape = _find_timer_shape(image,rect,line_width,threshold,radius_scale)
        if shape is not None:
            return shape
    return None


def _find_timer_shape(image, rect, line_width, outline_threshold, radius_scale):
    from .vision import thin, trace, RecognitionError
    x, y, w, h = rect
    pad = round(line_width)
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(image.shape[1], x+w+pad), min(image.shape[0], y+h+pad)
    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = ((hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 110) & (hsv[:, :, 2] < 205)).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gray)
    if count < 2:
        return None
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[label, cv2.CC_STAT_AREA] < line_width ** 2 * 5:
        return None
    core = (labels == label).astype(np.uint8) * 255
    # Resizing can dim the thin outline; keep its antialiased fragments while
    # excluding the darker gray shaft. Extents still come from actual pixels.
    white = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > outline_threshold)).astype(np.uint8) * 255
    radius = max(1, round(line_width * radius_scale))
    grouped = cv2.dilate(white, np.ones((radius*2+1, radius*2+1), np.uint8))
    count, groups = cv2.connectedComponents(grouped)
    whites = []
    for label in range(1, count):
        yy, xx = np.nonzero((groups == label) & (white > 0))
        if len(xx):
            whites.append((xx.min(), yy.min(), xx.max()-xx.min()+1, yy.max()-yy.min()+1, len(xx)))
    candidates = []
    for hx, hy, hw, hh, area in whites:
        if area < line_width ** 2 or min(hw, hh) < line_width or max(hw, hh) < line_width * 3:
            continue
        body = core.copy()
        body[hy:hy+hh, hx:hx+hw] = 0
        count, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body)
        if count < 2:
            continue
        body = (body_labels == 1 + np.argmax(body_stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)*255
        skeleton = thin(body)
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        degree = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = np.argwhere(skeleton & (degree == 1))[:, ::-1]
        center = np.array((hx + (hw-1)/2, hy + (hh-1)/2))
        near = endpoints[np.linalg.norm(endpoints - center, axis=1) < max(hw, hh) * .8]
        if not len(near):
            continue
        anchor = near.mean(axis=0)
        vector = center - anchor
        norm = np.linalg.norm(vector)
        if norm < line_width:
            continue
        direction = max((d for d,v in DIRECTIONS.items() if 0 in v), key=lambda d: np.dot(vector / norm, DIRECTIONS[d]))
        if np.dot(vector / norm, DIRECTIONS[direction]) < .85:
            continue
        # Physical head is identified by the sprite, not by an arbitrary body endpoint.
        dx, dy = DIRECTIONS[direction]
        tip = center.copy()
        if dx: tip[0] = hx + hw - 1 if dx > 0 else hx
        if dy: tip[1] = hy + hh - 1 if dy > 0 else hy
        try:
            path = trace(body, anchor + np.array((dx, dy)) * line_width * .7, direction, line_width)
        except RecognitionError:
            continue
        neck = anchor.copy()
        neck[0 if dy else 1] = tip[0 if dy else 1]
        # Complete the body segment hidden beneath the bullet sprite, in pixel steps.
        bridge = np.linspace(path[-1], neck, max(2, int(np.linalg.norm(path[-1]-neck)) + 1))
        extension = np.linspace(neck, tip, max(2, int(np.linalg.norm(neck-tip)) + 1))
        path = np.concatenate((path, bridge[1:], extension[1:])) + (x0, y0)
        visual = (int(x), int(y), int(w), int(h))
        head = (int(x0+hx), int(y0+hy), int(hw), int(hh))
        # Include sprite extents, while not treating the entire bounding box as occupied.
        left, top = min(x, head[0]), min(y, head[1])
        right, bottom = max(x+w, head[0]+hw), max(y+h, head[1]+hh)
        candidates.append((path, direction, tuple(int(v) for v in tip+(x0,y0)),
                           (int(left), int(top), int(right-left), int(bottom-top)), head))
    return candidates[0] if len(candidates) == 1 else None


def read_countdown(image):
    """Suggest red digits in the top-right white timer panel. Never auto-confirms OCR."""
    h, w = image.shape[:2]
    x0, y0 = round(w*.7), round(h*.025)
    roi = image[y0:round(h*.16), x0:]
    b, g, r = cv2.split(roi.astype(float))
    red = ((r > 140) & (r > g*1.5) & (r > b*1.5)).astype(np.uint8)*255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    glyphs = []
    for i, (x,y,gw,gh,area) in enumerate(stats[1:], 1):
        if gh < max(8, h*.006) or gh > h*.035 or area < 15:
            continue
        border = roi[max(0,y-3):min(roi.shape[0],y+gh+3), max(0,x-3):min(roi.shape[1],x+gw+3)]
        if (border.min(axis=2) > 200).mean() < .15:
            continue
        glyphs.append((x, y, (labels[y:y+gh,x:x+gw] == i).astype(np.uint8)*255))
    if not 1 <= len(glyphs) <= 3:
        return None, "请填写截图时的子弹倒计时"
    glyphs.sort(key=lambda p:p[0])
    def normalize(mask):
        yy,xx = np.nonzero(mask)
        if not len(xx):return np.zeros((48,32),bool)
        return cv2.resize(mask[yy.min():yy.max()+1,xx.min():xx.max()+1],(32,48)) > 100
    templates = []
    for digit in range(10):
        for font in (cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_TRIPLEX):
            for thick in (2,3,4):
                mask=np.zeros((70,65),np.uint8)
                cv2.putText(mask,str(digit),(8,55),font,1.6,255,thick,cv2.LINE_AA)
                templates.append((digit,normalize(mask)))
    digits=[];scores=[]
    for _,_,glyph in glyphs:
        target=normalize_digit(glyph)
        matches=[(float((target&t).sum()/max(1,(target|t).sum())),d) for d,t in templates]
        score,digit=max(matches)
        digits.append(str(digit));scores.append(score)
    if min(scores) < .5:
        return None,"倒计时数字不确定，请根据截图填写"
    seconds=int(''.join(digits))
    return float(seconds),f"截图数字候选 {seconds} 秒，请核对"


def find_wide_shape(mask, line_width):
    """Recover a straight decorated broad arrow from its filled outer silhouette."""
    from .vision import find_head, RecognitionError
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    if dt.max()*2 < line_width*3:
        return None
    try:
        tip, direction, quality = find_head(filled, line_width)
    except RecognitionError:
        return None
    dx, dy = DIRECTIONS[direction]
    if dx and dy:
        return None
    yy, xx = np.nonzero(filled)
    along, across = (yy, xx) if dy else (xx, yy)
    axis = 1 if dy else 0
    sign = dy or dx
    tail = along.min() if sign > 0 else along.max()
    length = abs(tip[axis]-tail)
    widths, centers = [], []
    for value in range(int(along.min()), int(along.max())+1):
        row = across[along == value]
        if len(row):
            widths.append(np.ptp(row)+1)
            centers.append((row.min()+row.max())/2)
    body_width = float(np.median(widths))
    if body_width < 2.5*line_width or length < body_width*3:
        return None
    center = float(np.median(centers))
    if np.quantile(np.abs(np.array(centers)-center), .95) > line_width*.5:
        return None  # Bent broad arrows need a separate, validated model.
    start = np.array((center, tail) if dy else (tail, center), float)
    tip[1-axis] = center
    path = np.linspace(start, tip, max(2, round(length)+1))
    click = tip-np.array((dx,dy))*body_width*.6
    return path, direction, tuple(np.rint(click).astype(int)), filled


def find_portals(image, crop):
    """Dark circular core surrounded by both red/orange and yellow spiral light."""
    x,y,w,h = map(int,crop)
    roi=image[y:y+h,x:x+w]
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    dark=(hsv[:,:,2]<65).astype(np.uint8)
    _,_,stats,centers=cv2.connectedComponentsWithStats(dark)
    found=[]
    for (bx,by,bw,bh,area),center in zip(stats[1:],centers[1:]):
        if min(bw,bh)<4 or max(bw,bh)>min(w,h)*.08 or not .75<bw/bh<1.33:
            continue
        if not .5<area/(bw*bh)<.92:
            continue
        radius=max(bw,bh)*1.5
        cx,cy=center
        left,top=max(0,int(cx-radius)),max(0,int(cy-radius))
        right,bottom=min(w,int(cx+radius)+1),min(h,int(cy+radius)+1)
        region=hsv[top:bottom,left:right]
        yy,xx=np.mgrid[top:bottom,left:right]
        distance=np.hypot(xx-cx,yy-cy)
        ring=(distance>max(bw,bh)*.55)&(distance<radius)
        warm=(region[:,:,1]>120)&(region[:,:,2]>170)
        yellow=warm&(region[:,:,0]>=15)&(region[:,:,0]<=40)
        red=warm&((region[:,:,0]<15)|(region[:,:,0]>170))
        white=(region[:,:,1]<70)&(region[:,:,2]>215)
        if not ring.any() or yellow[ring].mean()<.04 or red[ring].mean()<.12 or white[ring].mean()<.015:
            continue
        found.append(((float(cx+x),float(cy+y)),(left+x,top+y,right-left,bottom-top)))
    return found


def normalize_digit(mask):
    yy,xx=np.nonzero(mask)
    cropped = mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
    return cv2.resize((cropped > 0).astype(np.uint8)*255,(32,48))>100


@lru_cache(maxsize=1)
def counter_digit_templates():
    templates=[]
    for digit in range(10):
        for font in (cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_TRIPLEX):
            for thickness in (2,3,4,5):
                mask=np.zeros((70,65),np.uint8)
                cv2.putText(mask,str(digit),(8,55),font,1.6,255,thickness,cv2.LINE_AA)
                templates.append((digit,normalize_digit(mask>100)))
    # Small outlined game glyphs differ from Hershey fonts, especially 1 and 6.
    samples=json.loads((Path(__file__).parent/'assets/counter_digits.json').read_text())
    for sample in samples['glyphs']:
        mask=np.array([[int(pixel) for pixel in row] for row in sample['rows']],np.uint8)
        templates.append((sample['digit'],normalize_digit(mask)))
    return templates


def read_white_number(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    white=((hsv[:,:,1]<65)&(hsv[:,:,2]>205)).astype(np.uint8)*255
    n,labels,stats,_=cv2.connectedComponentsWithStats(white)
    glyphs=[]
    for idx,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h>=image.shape[0]*.35 and area>=3:
            glyphs.append((int(x),(labels[y:y+h,x:x+w]==idx).astype(np.uint8)*255))
    if not 1<=len(glyphs)<=3:
        return None
    templates = counter_digit_templates()
    digits=[]
    for _,glyph in sorted(glyphs,key=lambda p:p[0]):
        target=normalize_digit(glyph)
        score,digit=max((float((target&t).sum()/max(1,(target|t).sum())),d) for d,t in templates)
        if score<.48:
            return None
        digits.append(str(digit))
    return int(''.join(digits))


def find_counter_blocks(image,crop):
    x,y,w,h=map(int,crop)
    roi=image[y:y+h,x:x+w]
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    orange=((hsv[:,:,0]>12)&(hsv[:,:,0]<35)&(hsv[:,:,1]>120)&(hsv[:,:,2]>160)).astype(np.uint8)*255
    contours,_=cv2.findContours(orange,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bounds=[]
    for contour in contours:
        bx,by,bw,bh=cv2.boundingRect(contour)
        if min(bw,bh)>=10 and .85<bw/bh<1.18 and cv2.contourArea(contour)>=bw*bh*.8:
            bounds.append((bx,by,bw,bh))
    # Red diamonds and black/white lettering may split orange into separate
    # fragments. Recover the compact tile from its whole foreground instance.
    from .vision import foreground_mask
    _,labels,stats,_=cv2.connectedComponentsWithStats(foreground_mask(roi))
    for bx,by,bw,bh,area in stats[1:]:
        if min(bw,bh)<10 or not .8<bw/bh<1.25 or area/(bw*bh)<.72:
            continue
        patch=roi[by:by+bh,bx:bx+bw]
        if (orange[by:by+bh,bx:bx+bw]>0).mean()<.12:
            continue
        patch_hsv=hsv[by:by+bh,bx:bx+bw]
        white=(patch_hsv[:,:,1]<65)&(patch_hsv[:,:,2]>205)
        if white.mean()<.02:
            continue
        bounds.append(tuple(map(int,(bx,by,bw,bh))))
    result=[]
    for bx,by,bw,bh in bounds:
        center=(float(x+bx+(bw-1)/2),float(y+by+(bh-1)/2))
        if any(np.linalg.norm(np.array(center)-old_center)<min(bw,bh)*.5 for old_center,_,_ in result):
            continue
        patch=roi[by:by+bh,bx:bx+bw]
        value=read_white_number(patch)
        # A numbered obstacle also has a dark glyph outline on its orange tile.
        if (patch.max(axis=2)<90).mean()<.08:
            continue
        result.append((center,(x+bx,y+by,bw,bh),value))
    return result
