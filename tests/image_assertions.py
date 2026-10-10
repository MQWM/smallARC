"""Offline image regression assertions; not part of the desktop execution path."""
import cv2
import numpy as np
from arrow_solver.executor import ScreenValidator
from arrow_solver.vision import foreground_mask
from arrow_solver.special import find_portals, read_white_number


class ScreenVerifier(ScreenValidator):

    def __init__(self, source, board):
        super().__init__(source, board)
        self.source = source.copy()

    def expected(self, removed: set[int]):
        expected = self.source[self.roi].copy()
        for aid in removed:
            expected[self.masks[aid]] = self.background
        return expected

    def burning_matches(self, actual, arrow):
        """Allow a changed tail prefix, but keep the head and unburnt suffix registered."""
        x,y,_,_ = self.board.crop_rect
        hx,hy,hw,hh = arrow.head_rect
        original = self.source[self.roi].astype(float)
        head_delta = np.linalg.norm(actual[hy-y:hy-y+hh,hx-x:hx-x+hw]-original[hy-y:hy-y+hh,hx-x:hx-x+hw],axis=2)
        if not head_delta.size or (head_delta > 45).mean() > .25:
            return False
        retained = []
        radius = max(1, round(self.board.line_width*.2))
        for cell in arrow.cells:
            px,py = map(round,self.board.pixel(cell))
            if hx-radius <= px <= hx+hw+radius and hy-radius <= py <= hy+hh+radius:
                continue
            px,py = px-x,py-y
            ys=slice(max(0,py-radius),min(actual.shape[0],py+radius+1))
            xs=slice(max(0,px-radius),min(actual.shape[1],px+radius+1))
            before,after = original[ys,xs],actual[ys,xs]
            strong = np.linalg.norm(before-self.background,axis=2) > 35
            if strong.any():
                retained.append((np.linalg.norm(after[strong]-before[strong],axis=1)<45).mean() > .5)
        vanished = [i for i,value in enumerate(retained) if not value]
        # One cell tolerance for flame/antialiasing at the burning front.
        return not vanished or sum(retained[:max(vanished)]) <= 1

    def matches(self, frame: np.ndarray, removed: set[int], burning: bool = False) -> bool:
        if frame.shape != self.source.shape:
            return False
        actual = frame[self.roi].astype(float)
        expected = self.expected(removed).astype(float)
        delta = np.linalg.norm(actual - expected, axis=2)
        # Burning changes a tail prefix visually, but never releases solver occupancy.
        dynamic = self.portal_mask.copy()
        successful=sum(a.id in removed and a.kind != "timer" for a in self.board.arrows)
        for block,mask in zip(self.board.counter_blocks,self.block_masks):
            remaining=block.remaining-successful
            bx,by,bw,bh=block.visual_rect
            if remaining>0:
                if read_white_number(frame[by:by+bh,bx:bx+bw]) != remaining:
                    return False
            elif (np.linalg.norm(actual[mask]-self.background,axis=1)>45).mean()>.35:
                return False
            dynamic |= mask
        if self.board.portals:
            actual_portals = find_portals(frame,self.board.crop_rect)
            if len(actual_portals) != len(self.board.portals):
                return False
            for portal in self.board.portals:
                if not any(np.linalg.norm(np.array(center)-portal.center) < min(self.board.grid_step)*.2
                           for center,_ in actual_portals):
                    return False
        timers = {a.id:a for a in self.board.arrows if a.kind == "timer"}
        if burning:
            radius = max(1, round(self.board.line_width*1.5))
            for aid,arrow in timers.items():
                if aid not in removed:
                    if not self.burning_matches(actual,arrow):
                        return False
                    dynamic |= cv2.dilate(self.masks[aid].astype(np.uint8),
                                          np.ones((radius*2+1,radius*2+1),np.uint8)) > 0
        # Other instances are still checked individually, even inside the flame margin.
        if ((delta > 45) & ~dynamic).mean() > .025:
            return False
        original = self.source[self.roi].astype(float)
        for aid, mask in self.masks.items():
            if aid in removed:
                # Revealed grid dots are not arrow pixels: measure retained original
                # foreground, rather than demanding a perfectly flat background.
                strong = mask & (np.linalg.norm(original-self.background,axis=2) > 60)
                if not strong.any():
                    strong = mask
                retained = np.linalg.norm(actual[strong]-original[strong],axis=1) < 35
                if retained.mean() > .18:
                    return False
                # A replacement/dialog is not a valid disappearance either.
                if (delta[mask] > 45).mean() > .35:
                    return False
            elif next(a for a in self.board.arrows if a.id==aid).kind == "electric" or any(
                    a.id==aid and a.kind=="normal" and a.visual_rect is not None for a in self.board.arrows):
                if (foreground_mask(frame[self.roi])[mask]>0).mean()<.7:
                    return False
            elif burning and aid in timers:
                continue
            elif (delta[mask] > 45).mean() > .12:
                return False
        return True

