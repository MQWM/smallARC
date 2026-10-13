"""Permanent paired gates, recognized by cyan arch and a white four-point star."""
import cv2
import numpy as np
from .models import TeleportGate


def find_gates(image,crop):
    x,y,w,h = crop
    hsv = cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv,(80,65,100),(115,255,255))
    _,_,stats,_ = cv2.connectedComponentsWithStats(cyan)
    found = []
    for bx,by,bw,bh,area in stats[1:]:
        if min(bw,bh)<8 or bh<12 or not .4<bw/bh<.9 or bw>image.shape[1]*.12 or area<bw*bh*.15:
            continue
        patch = hsv[by:by+bh,bx:bx+bw]
        white = cv2.inRange(patch,(0,0,205),(180,65,255))
        _,_,parts,centers = cv2.connectedComponentsWithStats(white)
        stars = [(s,c) for s,c in zip(parts[1:],centers[1:])
                 if s[4]>=max(6,bw*bh*.025) and s[2]>bw*.22 and s[3]>bh*.22
                 and s[0]>0 and s[1]>0 and s[0]+s[2]<bw and s[1]+s[3]<bh
                 and .25*bw<c[0]<.75*bw and .25*bh<c[1]<.8*bh]
        if len(stars)!=1:
            continue
        s,c = stars[0]
        # A narrow cross/star inside a blue body; neutral rectangular badges
        # and blue ordinary arrows must not be classified as teleport gates.
        if not .12<s[4]/(s[2]*s[3])<.65:
            continue
        deep_blue = (patch[:,:,0]>=100)&(patch[:,:,0]<=145)&(patch[:,:,1]>80)&(patch[:,:,2]<210)
        if deep_blue.mean()<.08:
            continue
        found.append(TeleportGate(len(found)+1,(float(x+bx+c[0]),float(y+by+c[1])),
                                  (int(x+bx),int(y+by),int(bw),int(bh))))
    # Only the confirmed single-pair skin is supported; never guess multi-pair links.
    if len(found)==2:
        found[0].partner_id,found[1].partner_id = found[1].id,found[0].id
    return found
