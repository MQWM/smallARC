from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np

from .executor import ExecutionError
from .vision import read_image, recognize


@dataclass
class NextLevelResult:
    image: np.ndarray
    captured_at: float


@lru_cache(maxsize=3)
def _template(name):
    return read_image(Path(__file__).parent / 'assets' / name)


def _match(image, name, top, bottom):
    h,w = image.shape[:2]
    template = _template(name)
    scale = w/1080
    template = cv2.resize(template,None,fx=scale,fy=scale)
    y0,y1 = round(h*top),round(h*bottom)
    roi = image[y0:y1]
    th,tw = template.shape[:2]
    if roi.shape[0]<th or w<tw:
        return None
    scores = cv2.matchTemplate(roi,template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y) = cv2.minMaxLoc(scores)
    if score<.90:
        return None
    return (x+tw//2,y0+y+th//2)


def next_button(image):
    """Require both the victory heading and the labelled next-level button."""
    if _match(image,'victory_title.png',.18,.32) is None:
        return None
    return _match(image,'next_level_button.png',.60,.85)


def _header(image):
    h,w = image.shape[:2]
    patch = image[round(h*.04):round(h*.075),round(w*.40):round(w*.73)]
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    neutral = hsv[:,:,1]<65
    bright = hsv[:,:,2]
    # The victory modal dims the same header: use relative brightness so
    # removing that dim layer does not look like a different level number.
    peak = np.percentile(bright[neutral],99) if neutral.any() else 255
    mask = (neutral & (bright>peak*.72)).astype(np.uint8)
    return cv2.resize(mask,(160,40),interpolation=cv2.INTER_NEAREST)>0


def new_board(image, previous_header):
    current = _header(image)
    title = cv2.cvtColor(_template('level_title.png'),cv2.COLOR_BGR2GRAY)//255
    score = cv2.minMaxLoc(cv2.matchTemplate(current.astype(np.uint8),title,cv2.TM_CCOEFF_NORMED))[1]
    if score<.82:
        return False
    union = current | previous_header
    if current.sum()<30 or not union.any():
        return False
    same = (current & previous_header).sum()/union.sum()
    if same>.97:
        return False
    try:
        board = recognize(image).board
    except ValueError:
        return False
    # Entry detection tolerates unknown skins; the continuous runner applies
    # stricter recognition and execution checks before sending arrow taps.
    return bool(board.arrows)


class NextLevel:
    def __init__(self,device,control,report=lambda kind,data:None,*,
                 interval=5.,timeout=120.,now=time.monotonic):
        self.device,self.control,self.report = device,control,report
        self.interval,self.timeout,self.now = interval,timeout,now

    def run(self):
        deadline = self.now()+self.timeout
        previous_header = None
        while self.now()<deadline:
            self.control.check()
            captured_at = self.now()
            image = self.device.screenshot()
            self.control.check()
            point = next_button(image)
            if previous_header is None:
                if point is not None:
                    previous_header = _header(image)
                    self.control.check()
                    self.device.tap(*point)
                    self.report('next_level_clicked',{'point':point})
                else:
                    self.report('next_level_waiting',{'message':'等待胜利页与下一关按钮'})
            elif point is None and new_board(image,previous_header):
                self.report('next_level_entered',{})
                return NextLevelResult(image,captured_at)
            else:
                self.report('next_level_waiting',{'message':'已点击，等待新关卡棋盘'})
            wake_at = min(deadline,captured_at+self.interval)
            self.control.wait(max(0.,wake_at-self.now()))
        raise ExecutionError(f'自动下一关等待超过 {self.timeout:g} 秒，已停止；请检查手机画面')
