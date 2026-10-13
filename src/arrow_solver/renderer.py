from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from .models import Board, Plan, DIRECTIONS
from .vision import write_image


def overlay(image, board: Board, plan: Plan | None = None, preview: int = -1):
    canvas = image.copy()
    scale = max(1., image.shape[1] / 900)
    if board.orbit_obstacle:
        o = board.orbit_obstacle
        x,y,w,h = map(round,o.track_rect)
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,180,255),max(1,round(scale)))
        cv2.circle(canvas,tuple(map(round,o.center)),round(o.radius),(0,180,255),max(1,round(scale)))
        cv2.putText(canvas,'CAPTURE',tuple(map(round,(o.center[0]+o.radius+4,o.center[1]))),cv2.FONT_HERSHEY_SIMPLEX,.4*scale,(0,180,255),max(1,round(scale)),cv2.LINE_AA)
        cv2.putText(canvas,'ORBIT',(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,.4*scale,(0,180,255),max(1,round(scale)),cv2.LINE_AA)
    for portal in board.portals:
        center = tuple(round(v) for v in portal.center)
        radius = round(max(portal.visual_rect[2:])*.5)
        cv2.circle(canvas,center,radius,(80,230,255),max(1,round(scale)))
        cv2.putText(canvas,"EXIT",(center[0]+radius,center[1]),cv2.FONT_HERSHEY_SIMPLEX,.4*scale,(80,230,255),max(1,round(scale)),cv2.LINE_AA)
    for gate in board.teleport_gates:
        x,y,w,h = gate.visual_rect
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(255,230,70),max(1,round(scale)))
        cv2.putText(canvas,f'TP{gate.id}->{gate.partner_id or "?"}',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,.4*scale,(255,230,70),max(1,round(scale)),cv2.LINE_AA)
    if board.teleport_gates and plan and 0<=preview<len(plan.order):
        from .channels import route
        counts = {c.id:c.remaining for c in board.turn_channels}
        for aid in plan.order[:preview]:
            for cid in plan.channel_uses.get(aid,[]):
                counts[cid]-=1
        arrow = next(a for a in board.arrows if a.id==plan.order[preview])
        segments,_,_ = route(board,arrow,counts)
        previous = None
        for a,b in segments:
            if previous is not None and np.linalg.norm(np.array(a)-previous)>1:
                # Dashes connect a teleport jump; only solid lines are flights.
                delta = np.array(a)-previous
                steps = max(1,int(np.linalg.norm(delta)/(8*scale)))
                for i in range(0,steps,2):
                    p = tuple(np.rint(previous+delta*i/steps).astype(int))
                    q = tuple(np.rint(previous+delta*min(i+1,steps)/steps).astype(int))
                    cv2.line(canvas,p,q,(170,170,170),max(1,round(scale)))
            visible,p,q = cv2.clipLine(board.crop_rect,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))
            if visible:
                tip_length = min(.2,12*scale/max(1.,np.linalg.norm(np.array(q)-p)))
                cv2.arrowedLine(canvas,p,q,(100,255,120),max(1,round(scale*2)),tipLength=tip_length)
            previous = np.array(b)
    for block in board.counter_blocks:
        x,y,w,h = block.visual_rect
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,255),max(1,round(scale)))
        cv2.putText(canvas,f"BLOCK {block.remaining}",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,.4*scale,(0,255,255),max(1,round(scale)),cv2.LINE_AA)
    order = {aid: i for i, aid in enumerate(plan.order)} if plan else {}
    for channel in board.turn_channels:
        points = np.rint(channel.pixel_path).astype(np.int32)
        cv2.polylines(canvas,[points],False,(255,230,70),max(2,round(scale*2)))
        x,y,_,_ = channel.visual_rect
        cv2.putText(canvas,f'TURN {channel.id}: {channel.remaining}',(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,.4*scale,(255,230,70),max(1,round(scale)),cv2.LINE_AA)
    if preview > 0:
        faded = np.zeros(canvas.shape[:2], np.uint8)
        for arrow in board.arrows:
            if order.get(arrow.id, preview) < preview:
                points = np.rint(arrow.pixel_path or [board.pixel(c) for c in arrow.cells]).astype(np.int32)
                cv2.polylines(faded, [points], False, 255, max(3, round(board.line_width*1.8)))
        selected = faded > 0
        canvas[selected] = (canvas[selected]*.2 + 55*.8).astype(np.uint8)
    for arrow in board.arrows:
        points = np.rint(arrow.pixel_path or [board.pixel(c) for c in arrow.cells]).astype(np.int32)
        if arrow.kind == "wide":
            radius = max(2, round(min(board.grid_step)*.3))
            for cell in arrow.footprint:
                px,py = map(round,board.pixel(cell))
                cv2.rectangle(canvas,(px-radius,py-radius),(px+radius,py+radius),(230,210,90),max(1,round(scale)))
        current = preview >= 0 and order.get(arrow.id) == preview
        color = (70, 255, 255) if current else (255, 255, 255)
        cv2.polylines(canvas, [points], False, color, max(1, round(scale * (3 if current else 1))))
        head = points[-1]
        dx, dy = DIRECTIONS[arrow.direction]
        tip = tuple(np.rint(head + np.array((dx, dy)) * board.grid_step[0] * .35).astype(int))
        cv2.arrowedLine(canvas, tuple(head), tip, color, max(1, round(scale * 2)), tipLength=.5)
        text = str(order[arrow.id] + 1) if arrow.id in order else f"#{arrow.id}"
        if arrow.kind == "timer":
            text += " WAIT"
        elif arrow.kind == "wide":
            text += " WIDE"
        elif arrow.kind == "electric":
            text += " FLASH"
        pos = (int(head[0] + scale * 5), int(head[1] - scale * 7))
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .4 * scale, (20, 20, 20), max(3, round(4 * scale)), cv2.LINE_AA)
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .4 * scale, color, max(1, round(scale)), cv2.LINE_AA)
    return canvas


def export_run(root: Path, image, board: Board, plan: Plan | None):
    path = root / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    write_image(path / "source.png", image)
    board.save(path / "board.json")
    write_image(path / "overlay.png", overlay(image, board))
    if plan:
        arrows = {a.id: a for a in board.arrows}
        data = {"schema_version": 2, **asdict(plan),
                "timer_seconds_at_capture": board.timer_seconds,
                "portals": [asdict(p) for p in board.portals],
                "counter_blocks": [asdict(b) for b in board.counter_blocks],
                "turn_channels": [asdict(c) for c in board.turn_channels],
                "teleport_gates": [asdict(g) for g in board.teleport_gates],
                "orbit_obstacle": asdict(board.orbit_obstacle) if board.orbit_obstacle else None,
                "actions": [{"arrow": aid, "action": "wait_timer" if arrows[aid].kind == "timer" else "tap",
                             "point": None if arrows[aid].kind == "timer" else arrows[aid].click_point} for aid in plan.order],
                "clicks": [{"arrow": aid, "point": arrows[aid].click_point} for aid in plan.order if arrows[aid].kind != "timer"]}
        (path / "solution.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        write_image(path / "solution.png", overlay(image, board, plan))
    return path
