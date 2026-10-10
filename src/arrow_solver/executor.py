from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .models import Board, Plan
from .solver import verify_plan, block_threshold
from .vision import foreground_mask
from .special import timer_mask, timer_separated_components


class ExecutionError(RuntimeError):
    pass


class Cancelled(ExecutionError):
    pass


class ExecutionControl:
    def __init__(self):
        self.cancel = threading.Event()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("已停止；未再发送后续点击")

    def wait(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            self.cancel.wait(min(.05, max(0, deadline - time.monotonic())))
        self.check()


class ScreenValidator:
    """Validate source image instances and click points before sending any taps."""
    def __init__(self, source: np.ndarray, board: Board):
        self.board = board
        x, y, w, h = board.crop_rect
        self.roi = (slice(y, y + h), slice(x, x + w))
        roi = source[self.roi]
        fg = foreground_mask(roi)
        self.background = np.median(roi[fg == 0], axis=0) if np.any(fg == 0) else np.median(roi.reshape(-1, 3), axis=0)
        _, labels, _, _ = timer_separated_components(roi, fg, board.line_width)
        counts = np.bincount(labels.ravel())
        self.masks = {}
        assigned = set()
        self.portal_mask = np.zeros((h,w),bool)
        for portal in board.portals:
            px,py,pw,ph = portal.visual_rect
            self.portal_mask[py-y:py-y+ph,px-x:px-x+pw] = True
        assigned.update(int(v) for v in np.unique(labels[self.portal_mask]) if v)
        self.block_masks=[]
        for channel in board.turn_channels:
            cx,cy,cw,ch = channel.visual_rect
            region = labels[max(0,cy-y-2):min(h,cy-y+ch+2),max(0,cx-x-2):min(w,cx-x+cw+2)]
            overlaps = np.bincount(region.ravel(),minlength=len(counts))
            assigned.update(int(v) for v in np.flatnonzero(overlaps>counts*.5) if v)
        for block in board.counter_blocks:
            bx,by,bw,bh=block.visual_rect
            mask=np.zeros((h,w),bool)
            mask[max(0,by-y-2):by-y+bh+2,max(0,bx-x-2):bx-x+bw+2]=True
            self.block_masks.append(mask)
            overlaps = np.bincount(labels[mask], minlength=len(counts))
            # Padding may touch the tip of a neighboring arrow. Only claim
            # components predominantly inside the tile, not every touched label.
            assigned.update(int(v) for v in np.flatnonzero(overlaps > counts*.5) if v)
        for arrow in board.arrows:
            if arrow.kind == "timer":
                if not arrow.visual_rect or not arrow.head_rect:
                    raise ExecutionError(f"子弹 {arrow.id} 缺少完整视觉区域")
                mask = timer_mask(source, arrow.visual_rect, arrow.head_rect)[self.roi] > 0
                self.masks[arrow.id] = mask
                assigned.update(int(v) for v in np.unique(labels[mask]) if v)
                continue
            px, py = arrow.click_point[0] - x, arrow.click_point[1] - y
            if not (0 <= px < w and 0 <= py < h):
                raise ExecutionError(f"箭头 {arrow.id} 点击点越界")
            label = int(labels[py, px])
            if label == 0 or label in assigned:
                raise ExecutionError(f"箭头 {arrow.id} 的点击点不属于独立前景，请调整点击点")
            assigned.add(label)
            mask = labels == label
            decorated = arrow.kind == "normal" and arrow.visual_rect is not None
            if arrow.kind == "wide" or decorated:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(fg)
                cv2.drawContours(filled, contours, -1, 255, -1)
                mask = filled > 0
                assigned.update(int(v) for v in np.unique(labels[mask]) if v)
            # A chosen point must identify the recognized path, not a neighboring instance.
            good = 0
            points = [board.pixel(cell) for cell in arrow.cells]
            if arrow.launch_vectors or board.turn_channels:
                points = []
                for start,end in zip(arrow.pixel_path,arrow.pixel_path[1:]):
                    points.extend(np.linspace(start,end,max(2,int(np.linalg.norm(np.array(end)-start)/max(1,board.line_width*.5))+1)))
            for cx, cy in points:
                cx, cy = int(round(cx)) - x, int(round(cy)) - y
                if 0 <= cx < w and 0 <= cy < h and (mask[cy,cx] if decorated else labels[cy, cx] == label):
                    good += 1
            if good < max(1, len(points) * .8):
                raise ExecutionError(f"箭头 {arrow.id} 的路径与点击区域不一致")
            self.masks[arrow.id] = mask
        # Deleting an overlooked real instance must not make a partial board executable.
        minimum_area = max(12, w * h * .000015)
        unexplained = [label for label in range(1, len(counts))
                       if label not in assigned and counts[label] >= minimum_area]
        if unexplained:
            raise ExecutionError(f"棋盘还有 {len(unexplained)} 个未解释的前景区域，请补全箭头或调整裁剪")


class Executor:
    """Send a complete frozen plan without further screenshots."""
    def __init__(self, device, board: Board, plan: Plan, source, run_dir: Path,
                 control: ExecutionControl, report=lambda kind, data: None,
                 delay: float = .35, *, timer_deadline: float | None = None,
                 timer_duration: float | None = None, flight_wait: float = 5.,
                 not_before: float = 0., counter_settle: float = 2., now=time.monotonic):
        verify_plan(board,plan)
        if board.issues:
            raise ExecutionError("存在未解决的识别问题，不能自动执行")
        ScreenValidator(source,board)
        self.device,self.board,self.plan = device,board,plan
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True,exist_ok=True)
        self.control,self.report,self.delay = control,report,delay
        self.arrows = {a.id:a for a in board.arrows}
        self.timers = {a.id for a in board.arrows if a.kind=='timer'}
        self.timer_deadline,self.timer_duration = timer_deadline,timer_duration
        self.flight_wait,self.not_before,self.now = flight_wait,not_before,now
        self.counter_settle = max(0.,counter_settle)
        self.thresholds = {a.id:plan.counter_thresholds.get(a.id,block_threshold(board,a)) for a in board.arrows}
        self.counter_ready_at = {}
        self.urgent = set()
        pending = list(self.timers)
        while pending:
            for aid in plan.blockers[pending.pop()]:
                if aid not in self.urgent:
                    self.urgent.add(aid)
                    pending.append(aid)

    def log(self,kind,**data):
        entry = {"at":datetime.now(timezone.utc).isoformat(),"kind":kind,
                 "device":self.device.serial,**data}
        with (self.run_dir/'execution.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        self.report(kind,data)

    def check_budget(self,removed):
        if not self.timers:
            return
        if self.timer_deadline is None and self.timer_duration is None:
            raise ExecutionError("请确认子弹倒计时与是否已启动")
        pending = [aid for aid in self.urgent-removed if self.arrows[aid].kind!='timer']
        successes = sum(self.arrows[aid].kind!='timer' for aid in removed)
        unlock = max((block_threshold(self.board,self.arrows[aid]) for aid in
                      (self.urgent|self.timers)-removed),default=0)
        required = max(len(pending),unlock-successes)
        estimate = required*(self.delay+.25)
        # Counter-dependent missile exits need their animation margin within
        # the same deadline; waiting must not silently consume the clearing budget.
        timer_end = max((i for i,aid in enumerate(self.plan.order)
                         if aid in self.timers-removed),default=-1)
        gates = {self.thresholds[aid] for aid in self.plan.order[:timer_end+1]
                 if aid not in removed and self.thresholds[aid]>0}
        estimate += sum(max(0.,self.counter_ready_at[t]-self.now()) if t in self.counter_ready_at
                        else self.counter_settle for t in gates)
        remaining = self.timer_deadline-self.now() if self.timer_deadline is not None else self.timer_duration
        if estimate>0 and remaining < estimate+1.:
            raise ExecutionError(f"子弹清障时间不足：剩余约 {remaining:.1f} 秒，{required} 步预计 {estimate:.1f} 秒；已停止")
        self.report('budget',{'remaining':remaining,'steps':required,'estimate':estimate})

    def run(self,start: int=0):
        try:
            return self._run(start)
        except Exception as exc:
            self.log('halted',reason=str(exc))
            raise

    def _run(self,start):
        self.control.check()
        if not 0 <= start < len(self.plan.order):
            raise ExecutionError('执行游标无效')
        removed = set(self.plan.order[:start])
        sent_clicks = sum(self.arrows[aid].kind!='timer' for aid in removed)
        # Resume has only the last dispatch timestamp. Use that as a conservative
        # upper bound for all prior count milestones; never repeat their taps.
        last_sent = self.not_before-self.delay if self.not_before>0 else self.now()
        self.counter_ready_at = {t:last_sent+self.counter_settle
                                 for t in set(self.thresholds.values()) if 0<t<=sent_clicks}
        self.check_budget(removed)
        self.control.wait(max(0.,self.not_before-self.now()))
        self.log('ready',step=start,mode='batch_no_capture')
        for index in range(start,len(self.plan.order)):
            self.control.check()
            self.check_budget(removed)
            aid = self.plan.order[index]
            threshold = self.thresholds[aid]
            if threshold:
                wait = max(0.,self.counter_ready_at[threshold]-self.now())
                if wait:
                    self.log('waiting_counter',step=index,arrow=aid,threshold=threshold,seconds=wait)
                    self.control.wait(wait)
                    self.check_budget(removed)
            if aid in self.timers:
                if self.timer_deadline is None:
                    raise ExecutionError('倒计时尚未启动，当前步骤需要等待子弹；请手动启动游戏后重新抓图，并选择倒计时已启动')
                wait = max(0.,self.timer_deadline+self.flight_wait-self.now())
                self.log('waiting_timer',step=index,arrow=aid,seconds=wait)
                self.control.wait(wait)
                self.log('assumed_departed',step=index+1,arrow=aid,total=len(self.plan.order))
            else:
                self.log('clicking',step=index,arrow=aid)
                self.control.check()
                self.log('attempted',step=index,arrow=aid)
                if self.timers and self.timer_deadline is None:
                    self.timer_deadline = self.now()+self.timer_duration
                    self.log('timer_started',deadline=self.timer_deadline,duration=self.timer_duration)
                self.device.tap(*self.arrows[aid].click_point)
                sent_clicks += 1
                if sent_clicks in self.thresholds.values():
                    self.counter_ready_at[sent_clicks] = self.now()+self.counter_settle
                self.log('sent',step=index,arrow=aid)
                self.not_before = self.now()+self.delay
                uses = self.plan.channel_uses.get(aid,[])
                if uses:
                    # A lane consumes a passage only after the whole arrow exits.
                    # Keep its next routing state out of use during that flight.
                    self.not_before = max(self.not_before,self.now()+self.counter_settle)
                    self.log('waiting_channel',step=index,arrow=aid,channels=uses,
                             seconds=max(0.,self.not_before-self.now()))
                self.log('dispatched',step=index+1,arrow=aid,total=len(self.plan.order),not_before=self.not_before)
                self.control.wait(max(0.,self.not_before-self.now()))
            removed.add(aid)
        self.log('sent_complete',step=len(self.plan.order),mode='batch_no_capture')
        return len(self.plan.order)
