from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path

Cell = tuple[int, int]  # column, row
DIRECTIONS: dict[str, Cell] = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0), "UP_RIGHT": (1,-1), "DOWN_RIGHT": (1,1), "DOWN_LEFT": (-1,1), "UP_LEFT": (-1,-1)}


@dataclass
class Arrow:
    id: int
    cells: list[Cell]  # tail to head, every intermediate cell included
    direction: str
    click_point: tuple[int, int]
    quality: float = 1.0
    color: tuple[int, int, int] = (80, 210, 170)  # RGB
    kind: str = "normal"  # normal: tap; timer: clear blockers, then wait for departure
    visual_rect: tuple[int, int, int, int] | None = None
    head_rect: tuple[int, int, int, int] | None = None

    footprint: list[Cell] = field(default_factory=list)  # physical occupancy override; cells remains centerline
    pixel_path: list[tuple[float, float]] = field(default_factory=list)  # original-image polyline, tail to head
    launch_vectors: list[tuple[float, float]] = field(default_factory=list)  # non-grid geometry; all possible exits must clear

    @property
    def head(self) -> Cell:
        return self.cells[-1]


@dataclass
class Portal:
    cell: Cell
    center: tuple[float, float]
    visual_rect: tuple[int, int, int, int]


@dataclass
class CounterBlock:
    cell: Cell
    remaining: int
    visual_rect: tuple[int,int,int,int]


@dataclass
class TurnChannel:
    # Canonical orientation; reverse entry follows the reversed pixel_path.
    id: int
    remaining: int | None
    entry: tuple[float,float]
    exit: tuple[float,float]
    entry_direction: str
    exit_direction: str
    pixel_path: list[tuple[float,float]]
    visual_rect: tuple[int,int,int,int]


@dataclass
class Board:
    image_width: int
    image_height: int
    crop_rect: tuple[int, int, int, int]  # x, y, width, height
    grid_origin: tuple[float, float]  # ORIGINAL image pixels
    grid_step: tuple[float, float]
    cols: int
    rows: int
    arrows: list[Arrow]
    source_hash: str = ""
    issues: list[str] = field(default_factory=list)
    revision: int = 1
    line_width: float = 8.0
    timer_seconds: float | None = None  # value shown at capture, never a fresh countdown
    timer_reading: str = ""  # suggested OCR or manual; execution requires user confirmation

    portals: list[Portal] = field(default_factory=list)
    counter_blocks: list[CounterBlock] = field(default_factory=list)
    turn_channels: list[TurnChannel] = field(default_factory=list)

    def pixel(self, cell: Cell) -> tuple[float, float]:
        return tuple(self.grid_origin[i] + cell[i] * self.grid_step[i] for i in range(2))

    def cell(self, point: tuple[float, float]) -> Cell:
        return tuple(round((point[i] - self.grid_origin[i]) / self.grid_step[i]) for i in range(2))

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Board:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["arrows"] = [Arrow(**{**a, "cells": [tuple(c) for c in a["cells"]],
                                    "click_point": tuple(a["click_point"]), "color": tuple(a["color"]),
                                    "footprint": [tuple(c) for c in a.get("footprint", [])],
                                    "pixel_path": [tuple(p) for p in a.get("pixel_path", [])],
                                    "launch_vectors": [tuple(v) for v in a.get("launch_vectors", [])]})
                          for a in data["arrows"]]
        data["portals"] = [Portal(tuple(p["cell"]), tuple(p["center"]), tuple(p["visual_rect"])) for p in data.get("portals", [])]
        data["counter_blocks"] = [CounterBlock(tuple(p["cell"]),p["remaining"],tuple(p["visual_rect"])) for p in data.get("counter_blocks",[])]
        data['turn_channels'] = [TurnChannel(**{**c,'entry':tuple(c['entry']),'exit':tuple(c['exit']),
                                      'pixel_path':[tuple(p) for p in c['pixel_path']],
                                      'visual_rect':tuple(c['visual_rect'])}) for c in data.get('turn_channels',[])]
        for key in ("crop_rect", "grid_origin", "grid_step"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class Plan:
    order: list[int]
    blockers: dict[int, list[int]]
    board_fingerprint: str
    channel_uses: dict[int,list[int]] = field(default_factory=dict)
    counter_thresholds: dict[int,int] = field(default_factory=dict)


def image_hash(image) -> str:
    return sha256(image.tobytes()).hexdigest()
