from __future__ import annotations

from copy import deepcopy
import json
import time
from pathlib import Path
import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QMessageBox, QSplitter, QListWidget, QPlainTextEdit,
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QScrollArea)

from ..device import AndroidDevice, Mirror, list_devices, tool_path
from ..executor import Executor, ExecutionControl
from ..next_level import NextLevel, NextLevelResult
from ..continuous import ContinuousPlay
from ..models import DIRECTIONS
from ..renderer import overlay, export_run
from ..solver import solve
from ..vision import recognize, suggest_crop
from .board_view import BoardView
from .workers import Task
from .path_editor import edited_arrow, snapped_point
from ..geometry import point_distance



class MainWindow(QMainWindow):
    def __init__(self, run_root: Path):
        super().__init__()
        self.setWindowTitle("SmallARC · 小小箭头辅助工具")
        self.resize(1140, 880)
        self.adb = tool_path("adb")
        self.scrcpy = tool_path("scrcpy")
        self.run_root = run_root
        self.source = self.board = self.plan = self.crop = None
        self.source_device = None
        self.captured_at = None
        self.timer_deadline = None
        self.next_action_at = 0.
        self.cursor = 0
        self.preview = -1
        self.busy = False
        self.execution_active = False
        self.uncertain = False
        self.task = None
        self.control = None
        self.mirror = Mirror()
        self.closing = False
        self.editing = False
        self.edit_target = None
        self.edit_click = None
        self.edit_click_explicit = False
        self._build()
        self._update_controls()
        self.timer_tick = QTimer(self)
        self.timer_tick.timeout.connect(self.update_timer_text)
        self.timer_tick.start(1000)

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)
        self.device_box = QComboBox()
        self.device_box.setMinimumWidth(230)
        self.device_box.currentIndexChanged.connect(self._update_controls)
        toolbar.addWidget(self.device_box)
        self.refresh_button = self._button("刷新设备", self.refresh_devices, toolbar)
        self.mirror_button = self._button("启动投屏", self.start_mirror, toolbar)
        self.capture_button = self._button("获取截图", self.capture, toolbar)
        split = QSplitter()
        layout.addWidget(split, 1)
        self.view = BoardView()
        self.view.crop_selected.connect(self.set_crop)
        self.view.point_selected.connect(self.point_selected)
        self.view.point_hovered.connect(self.point_hovered)
        self.view.edit_finished.connect(self.save_arrow_edit)
        self.view.edit_cancelled.connect(self.cancel_arrow_edit)
        self.view.path_changed.connect(self.edit_path_changed)
        self.view.click_point_changed.connect(self.edit_click_changed)
        split.addWidget(self.view)
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(480)
        side = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        split.addWidget(scroll)
        split.setSizes([720, 380])
        prepare = QHBoxLayout()
        side.addLayout(prepare)
        self.crop_button = self._button("框选棋盘", self.view.begin_crop, prepare)
        self.recognize_button = self._button("识别关卡", self.analyze, prepare)
        self.summary = QLabel("连接手机并获取截图。滚轮缩放，拖动平移。")
        self.summary.setWordWrap(True)
        side.addWidget(self.summary)
        options = QHBoxLayout()
        side.addLayout(options)
        self._button("适应窗口", self.view.fit_image, options)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select_preview)
        self.list.setMinimumHeight(120)
        side.addWidget(self.list, 1)
        edits = QHBoxLayout()
        side.addLayout(edits)
        self.edit_button = self._button("编辑箭头", self.edit_arrow, edits)
        self.add_button = self._button("补画箭头", lambda: self.edit_arrow(new=True), edits)
        self.delete_button = self._button("删除箭头", self.delete_arrow, edits)
        self.edit_panel = QGroupBox("在图上编辑")
        edit_layout = QVBoxLayout(self.edit_panel)
        self.edit_note = QLabel("按尾部 → 各拐点 → 头部点击；拖动圆点调整。Enter 保存，Esc 取消，右键撤销末点。")
        self.edit_note.setWordWrap(True)
        edit_layout.addWidget(self.edit_note)
        self.edit_direction = QComboBox()
        for label,d in zip(('↑','↓','←','→','↗','↘','↙','↖'),DIRECTIONS):
            self.edit_direction.addItem(label,d)
        edit_layout.addWidget(self.edit_direction)
        actions = QHBoxLayout()
        edit_layout.addLayout(actions)
        self._button("设置点击点", self.pick_arrow_click, actions)
        self._button("撤销末点", self.view.undo_vertex, actions)
        finish = QHBoxLayout()
        edit_layout.addLayout(finish)
        self._button("保存", self.save_arrow_edit, finish)
        self._button("取消", self.cancel_arrow_edit, finish)
        side.addWidget(self.edit_panel)
        self.edit_panel.hide()
        self.solve_button = self._button("重新求解", self.recompute, side)
        self.reviewed = QCheckBox("已逐条核对原图、方向和路径（启用点击）")
        self.reviewed.setToolTip("完整计划有解并不能证明识别正确。确认后才可执行。")
        self.reviewed.toggled.connect(self.review_changed)
        side.addWidget(self.reviewed)
        execution = QGroupBox("整关执行 · 手动操作游戏前请先停止")
        exec_layout = QVBoxLayout(execution)
        self.mode_note = QLabel("一次截图识别后按顺序执行，中途不抓图；进度表示点击已发送。")
        self.mode_note.setWordWrap(True)
        exec_layout.addWidget(self.mode_note)
        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("最短动画等待（秒）"))
        self.delay = QDoubleSpinBox()
        self.delay.setRange(.1, 10)
        self.delay.setValue(.4)
        self.delay.setSingleStep(.1)
        delay_row.addWidget(self.delay)
        exec_layout.addLayout(delay_row)
        self.orbit_group = QGroupBox("小豆豆避让 · 估计位置，中途不抓图")
        orbit_form = QFormLayout(self.orbit_group)
        self.orbit_speed = QDoubleSpinBox()
        self.orbit_speed.setRange(1.,3000.)
        self.orbit_speed.setValue(118.)
        self.orbit_speed.setSuffix(" px/s")
        self.orbit_speed.setToolTip("手机原始截图坐标中的像素/秒；地图变大时绕圈时间自动变长。")
        orbit_form.addRow("小豆豆速度",self.orbit_speed)
        self.orbit_direction = QComboBox()
        self.orbit_direction.addItems(["顺时针","逆时针"])
        orbit_form.addRow("移动方向",self.orbit_direction)
        self.orbit_window = QDoubleSpinBox()
        self.orbit_window.setRange(.5,15.)
        self.orbit_window.setValue(3.)
        self.orbit_window.setSuffix(" 秒")
        self.orbit_window.setToolTip("覆盖点击到箭头头部抵达外围轨道的时间；若仍碰撞可调大。另有体积、延迟与预测误差余量。")
        orbit_form.addRow("出口避让窗口",self.orbit_window)
        exec_layout.addWidget(self.orbit_group)
        start_row = QHBoxLayout()
        exec_layout.addLayout(start_row)
        self.run_button = self._button("执行整关 / 继续", self.execute, start_row)
        self.next_level_button = self._button("自动下一关", self.advance_level, exec_layout)
        self.auto_next = QCheckBox("自动下一关、识别并连续执行")
        self.auto_next.setToolTip("开启后，新关卡自动识别、求解并执行；识别或校验失败会停止。")
        exec_layout.addWidget(self.auto_next)
        stop_row = QHBoxLayout()
        exec_layout.addLayout(stop_row)
        self.stop_button = self._button("停止", self.stop, stop_row)
        self.progress_label = QLabel("尚未执行")
        self.progress_label.setWordWrap(True)
        exec_layout.addWidget(self.progress_label)
        self.timer_group = QGroupBox("倒计时子弹（不点击，优先清障）")
        timer_form = QFormLayout(self.timer_group)
        self.countdown = QDoubleSpinBox()
        self.countdown.setRange(-1,9999)
        self.countdown.setSpecialValueText("未识别")
        self.countdown.setValue(-1)
        self.countdown.valueChanged.connect(self.countdown_changed)
        timer_form.addRow("截图显示的秒数",self.countdown)
        self.flight_wait = QDoubleSpinBox()
        self.flight_wait.setRange(.1,60)
        self.flight_wait.setValue(5.)
        timer_form.addRow("到时后离场等待（秒）",self.flight_wait)
        self.timer_running = QCheckBox("抓图时倒计时已经启动")
        self.timer_running.toggled.connect(self.timer_running_changed)
        timer_form.addRow(self.timer_running)
        self.timer_confirmed = QCheckBox("已核对秒数及是否已启动")
        self.timer_confirmed.toggled.connect(self._update_controls)
        timer_form.addRow(self.timer_confirmed)
        self.timer_note = QLabel()
        self.timer_note.setWordWrap(True)
        timer_form.addRow(self.timer_note)
        exec_layout.addWidget(self.timer_group)
        self.timer_group.hide()
        side.addWidget(execution)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(500)
        self.log.setMaximumHeight(160)
        side.addWidget(self.log)
        self.coordinate_label = QLabel("像素坐标：—")
        self.statusBar().addPermanentWidget(self.coordinate_label)
        self.statusBar().showMessage("就绪 · 连接手机并获取截图")

    def selected_serial(self):
        return self.device_box.currentData()

    def message(self, text):
        self.log.appendPlainText(text)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
        self.statusBar().showMessage(text[:140])

    def _update_controls(self, *_):
        idle = not self.busy and not self.editing
        self.refresh_button.setEnabled(idle)
        self.device_box.setEnabled(idle)
        self.mirror_button.setEnabled(idle and bool(self.selected_serial()))
        self.capture_button.setEnabled(idle and bool(self.selected_serial()))
        for button in (self.crop_button, self.recognize_button):
            button.setEnabled(idle and self.source is not None)
        for button in (self.edit_button, self.add_button, self.delete_button, self.solve_button):
            button.setEnabled(idle and self.board is not None)
        self.reviewed.setEnabled(idle and self.board is not None)
        self.delay.setEnabled(idle)
        self.orbit_group.setVisible(self.board is not None and self.board.orbit_obstacle is not None)
        self.orbit_group.setEnabled(idle)
        self.countdown.setEnabled(idle and self.cursor == 0 and self.timer_deadline is None)
        self.timer_running.setEnabled(idle and self.cursor == 0 and self.timer_deadline is None)
        self.flight_wait.setEnabled(idle)
        self.timer_confirmed.setEnabled(idle)
        has_timer = self.board is not None and any(a.kind == "timer" for a in self.board.arrows)
        timer_ready = not has_timer or (self.timer_confirmed.isChecked() and self.countdown.value() >= 0)
        self.timer_group.setVisible(has_timer)
        self.update_timer_text()
        can_run = (idle and self.plan is not None and self.reviewed.isChecked() and
                   self.cursor < len(self.plan.order) and not self.uncertain and
                   bool(self.selected_serial()) and self.source_device == self.selected_serial() and timer_ready)
        self.run_button.setEnabled(bool(can_run))
        self.next_level_button.setEnabled(idle and bool(self.selected_serial()))
        self.auto_next.setEnabled(idle)
        self.stop_button.setEnabled(self.execution_active)

    def work(self, function, done, execution=False):
        if self.busy:
            return
        self.busy = True
        self.execution_active = execution
        task = Task(function, self)
        self.task = task
        task.succeeded.connect(done)
        task.failed.connect(self.failed)
        task.event.connect(self.execution_event)
        task.finished.connect(self.task_finished)
        self._update_controls()
        task.start()

    def task_finished(self):
        self.busy = False
        self.execution_active = False
        self.task.deleteLater()
        self.task = None
        self._update_controls()
        if self.closing:
            QTimer.singleShot(0, self.close)

    def failed(self, text):
        self.message(text)
        if self.execution_active and self.uncertain:
            self.message("点击已发送但未确认发送完成：请重新获取截图并识别后再执行。")

    def refresh_devices(self):
        adb = self.adb
        self.work(lambda _: list_devices(adb), self.devices_ready)

    def devices_ready(self, devices):
        selected = self.selected_serial()
        self.device_box.clear()
        for d in devices:
            self.device_box.addItem(f"{d.serial} · {d.state}", d.serial if d.state == "device" else None)
            if d.serial == selected:
                self.device_box.setCurrentIndex(self.device_box.count() - 1)
        self.message(f"发现 {len(devices)} 台设备；unauthorized 需在手机上授权" if devices else "未发现设备，请连接 USB 并开启 USB 调试")

    def start_mirror(self):
        try:
            self.mirror.start(self.selected_serial(), self.scrcpy)
            self.message("已请求启动 scrcpy，请将手机窗口和辅助窗口并排摆放")
            QTimer.singleShot(1800, self.check_mirror)
        except Exception as exc:
            self.message(str(exc))

    def check_mirror(self):
        if self.mirror.process and self.mirror.process.poll() is not None:
            self.message("scrcpy 已退出；请检查设备授权，或在终端运行 scrcpy 查看详细错误")

    def capture(self):
        serial, adb = self.selected_serial(), self.adb
        def shot(_):
            captured = time.monotonic()
            return AndroidDevice(serial, adb).screenshot(), captured
        self.work(shot, lambda result: self.set_source(result[0], serial, result[1]))

    def set_source(self, image, serial, captured_at=None):
        self.cancel_arrow_edit()
        self.source = image
        self.source_device = serial
        self.captured_at = captured_at if captured_at is not None else time.monotonic()
        self.next_action_at = 0.
        self.timer_deadline = None
        self.timer_running.blockSignals(True)
        self.timer_running.setChecked(False)
        self.timer_running.blockSignals(False)
        self.timer_confirmed.setChecked(False)
        self.countdown.blockSignals(True)
        self.countdown.setValue(-1)
        self.countdown.blockSignals(False)
        self.board = self.plan = None
        self.cursor = 0
        self.preview = -1
        self.uncertain = False
        self.reviewed.setChecked(False)
        self.list.clear()
        self.view.set_image(image, reset=True)
        try:
            self.crop = suggest_crop(image)
            self.summary.setText(f"{image.shape[1]}×{image.shape[0]} · 建议棋盘范围 {self.crop}\n可框选调整后识别。")
        except Exception:
            self.crop = (0, 0, image.shape[1], image.shape[0])
        self.view.show_crop(self.crop)
        self.progress_label.setText("尚未执行")
        self.message("已获取手机原始截图" if serial else "当前图像未绑定设备，请获取手机截图后执行")
        self._update_controls()

    def invalidate(self):
        self.plan = None
        self.cursor = 0
        self.uncertain = False
        self.reviewed.setChecked(False)
        if self.board:
            self.board.revision += 1
        self._update_controls()

    def set_crop(self, rect):
        if self.busy:
            return
        self.crop = rect
        self.view.show_crop(rect)
        self.invalidate()
        self.board = None
        self.list.clear()
        self.message(f"棋盘范围已设为 {rect}，请重新识别")
        self.summary.setText(f"棋盘范围：{rect}")
        self._update_controls()

    def analyze(self):
        if self.source is None:
            return
        self.invalidate()
        image, crop = self.source.copy(), self.crop
        self.work(lambda _: recognize(image, crop), self.recognized)
        self.message("正在分割箭头、定位头部并还原网格…")

    def recognized(self, result):
        self.board = result.board
        self.crop = self.board.crop_rect
        self.view.show_crop(self.crop)
        self.countdown.blockSignals(True)
        self.countdown.setValue(self.board.timer_seconds if self.board.timer_seconds is not None else -1)
        self.countdown.blockSignals(False)
        self.timer_confirmed.setChecked(False)
        self.summary.setText(f"识别 {len(self.board.arrows)} 条箭头 · {len(self.board.portals)} 个黑瞳 · {len(self.board.counter_blocks)} 个数字障碍 · {len(self.board.turn_channels)} 个转弯通道 · {self.board.cols} 列 × {self.board.rows} 行\n网格间距 {self.board.grid_step[0]:.2f} / {self.board.grid_step[1]:.2f} px")
        for block in self.board.counter_blocks:
            self.message(f"数字障碍 {block.cell}：再成功消除 {block.remaining} 支箭头后消失，请核对读数")
        for channel in self.board.turn_channels:
            self.message(f"转弯通道 {channel.id}：两端可进入，再通过 {channel.remaining} 支后消失，请核对读数和开口")
        if self.board.orbit_obstacle:
            self.message("检测到小豆豆轨道；默认顺时针 118 像素/秒，按出口时间窗口简单避让，执行不核验是否退回。")
        for issue in self.board.issues:
            self.message(issue)
        self.recompute()

    def timer_running_changed(self, *_):
        self.timer_confirmed.setChecked(False)
        self.update_timer_text()

    def countdown_changed(self, value):
        self.timer_confirmed.setChecked(False)
        if self.board:
            self.board.timer_seconds = value if value >= 0 else None
            self.board.timer_reading = "手动填写截图时倒计时"
            self.recompute()
        self.update_timer_text()

    def update_timer_text(self):
        if not self.board or not any(a.kind == "timer" for a in self.board.arrows):
            return
        if self.countdown.value() < 0:
            self.timer_note.setText("无法可靠读取，请填写截图时的秒数并核对。")
            return
        if not self.source_device:
            self.timer_note.setText(f"离线图：截图读数 {self.countdown.value():g} 秒；连接设备后需重新抓图。")
            return
        deadline = self.timer_deadline
        if deadline is None and self.timer_running.isChecked():
            deadline = self.captured_at+self.countdown.value()
        if deadline is None:
            self.timer_note.setText(f"尚未启动：首次点击游戏屏幕后开始 {self.countdown.value():g} 秒倒计时。")
        else:
            self.timer_note.setText(f"倒计时已启动，预计剩余 {max(0,deadline-time.monotonic()):.0f} 秒；继续执行不会重置。")

    def recompute(self):
        if self.board is None:
            return
        self.invalidate()
        try:
            self.plan = solve(self.board)
            self.message(f"已计算 {len(self.plan.order)} 步完整顺序；请先核对识别覆盖图")
        except Exception as exc:
            self.message(str(exc))
        self.populate()
        self.redraw()
        self._update_controls()

    def populate(self):
        self.list.blockSignals(True)
        self.list.clear()
        arrows = {a.id: a for a in self.board.arrows}
        ids = self.plan.order if self.plan else sorted(arrows)
        for i, aid in enumerate(ids):
            arrow = arrows[aid]
            label = f"第 {i + 1} 步" if self.plan else "待校验"
            action = "等待子弹离场" if arrow.kind == "timer" else "点击粗箭头" if arrow.kind == "wide" else "点击闪电箭头" if arrow.kind == "electric" else "点击箭头"
            self.list.addItem(f"{label} · {action} #{aid} · {arrow.direction} · {len(arrow.cells)} 格")
        self.list.blockSignals(False)
        self.preview = -1

    def selected_arrow(self):
        if not self.board:
            return None
        ids = self.plan.order if self.plan else sorted(a.id for a in self.board.arrows)
        row = self.list.currentRow()
        return next((a for a in self.board.arrows if 0 <= row < len(ids) and a.id == ids[row]), None)

    def select_preview(self, row):
        self.preview = row
        self.redraw()

    def redraw(self, *_):
        if self.source is not None:
            self.view.set_image(overlay(self.source, self.board, self.plan, self.preview)
                                if self.board else self.source)

    def point_hovered(self, point):
        if point is None:
            self.coordinate_label.setText("像素坐标：—")
            return
        text = f"原图像素：X={point[0]}，Y={point[1]}"
        if self.board:
            text += f" · 网格 {self.board.cell(point)}"
        self.coordinate_label.setText(text)

    def point_selected(self, point):
        if not self.board or self.busy:
            return
        cell = self.board.cell(point)
        self.statusBar().showMessage(f"原图坐标 ({point[0]:.0f}, {point[1]:.0f}) · 网格 {cell}")
        candidates = []
        for a in self.board.arrows:
            points = a.pixel_path or [self.board.pixel(c) for c in a.cells]
            distance = min((point_distance(point,p,q) for p,q in zip(points,points[1:])),default=point_distance(point,points[0],points[0]))
            if distance <= self.board.line_width*1.3:
                candidates.append((distance,a.id))
        if candidates:
            aid = min(candidates)[1]
            ids = self.plan.order if self.plan else sorted(a.id for a in self.board.arrows)
            self.list.setCurrentRow(ids.index(aid))
            if self.editing and self.view.edit_points is None:
                self.begin_arrow_path(self.selected_arrow())

    def edit_arrow(self, checked=False, new=False):
        if not self.board or self.busy:
            return
        self.editing = True
        self.edit_panel.show()
        self.redraw()
        current = None if new else self.selected_arrow()
        if new or current:
            self.begin_arrow_path(current)
        else:
            self.edit_note.setText("请在图上点击要编辑的箭头。")
        self._update_controls()

    def begin_arrow_path(self,current):
        self.edit_target = current
        self.edit_click = current.click_point if current else None
        self.edit_click_explicit = False
        points = current.pixel_path or [self.board.pixel(c) for c in current.cells] if current else []
        self.view.snap_point = lambda point: snapped_point(self.board,point)
        self.view.begin_path(points,self.edit_click)
        self.edit_direction.setCurrentIndex(list(DIRECTIONS).index(current.direction if current else 'RIGHT'))
        self.edit_note.setText("按尾部 → 各拐点 → 头部点击；拖动圆点调整。Enter 保存，Esc 取消，右键撤销末点。")

    def edit_path_changed(self,points):
        if len(points)>1:
            vector = np.array(points[-1])-points[-2]
            direction = max(DIRECTIONS,key=lambda d: np.dot(vector,np.array(DIRECTIONS[d])/np.linalg.norm(DIRECTIONS[d])))
            self.edit_direction.setCurrentIndex(list(DIRECTIONS).index(direction))
            original_head = (self.edit_target.pixel_path[-1] if self.edit_target.pixel_path else self.board.pixel(self.edit_target.head)) if self.edit_target else None
            if not self.edit_click_explicit and (original_head is None or tuple(points[-1]) != tuple(original_head)):
                self.edit_click = points[-1]
                self.view.edit_click_point = self.edit_click
                self.view.draw_path()

    def pick_arrow_click(self):
        if self.view.edit_points is not None:
            self.view.pick_click = True
            self.edit_note.setText("在箭头头部内点击，设置实际发送到手机的点击位置。")

    def edit_click_changed(self,point):
        self.edit_click = point
        self.edit_click_explicit = True
        self.edit_note.setText(f"点击点已设为 ({point[0]:.0f}, {point[1]:.0f})；可继续调整路径或保存。")

    def save_arrow_edit(self):
        if not self.editing or self.view.edit_points is None:
            return
        try:
            arrow = edited_arrow(self.board,self.view.edit_points,self.edit_target,self.edit_click,self.edit_direction.currentData())
            if self.edit_target:
                self.board.arrows[self.board.arrows.index(self.edit_target)] = arrow
            else:
                self.board.arrows.append(arrow)
        except ValueError as exc:
            self.edit_note.setText(str(exc))
            return
        self.cancel_arrow_edit()
        self.recompute()

    def cancel_arrow_edit(self):
        self.editing = False
        self.edit_target = None
        self.edit_click = None
        self.view.end_path()
        self.edit_panel.hide()
        self._update_controls()

    def delete_arrow(self):
        current = self.selected_arrow()
        if current:
            self.board.arrows.remove(current)
            self.recompute()

    def review_changed(self, checked):
        if checked and self.board and self.board.issues:
            answer = QMessageBox.question(self, "确认已修正识别问题", "识别曾报告以下问题：\n" + "\n".join(self.board.issues) + "\n\n是否已补全/修正所有相关箭头？")
            if answer != QMessageBox.StandardButton.Yes:
                self.reviewed.setChecked(False)
                return
            self.board.issues.clear()
            self.board.revision += 1
            try:
                self.plan = solve(self.board)
            except Exception as exc:
                self.plan = None
                self.message(str(exc))
                self.reviewed.setChecked(False)
        self._update_controls()

    def execute(self):
        if self.busy or not self.plan or not self.reviewed.isChecked() or self.uncertain:
            return
        serial = self.selected_serial()
        if not serial or self.source_device != serial:
            self.message("请先获取该手机的原始截图并重新识别")
            return
        self.control = ExecutionControl()
        board, plan, source = deepcopy(self.board), deepcopy(self.plan), self.source.copy()
        device = AndroidDevice(serial, self.adb)
        control, start, delay = self.control, self.cursor, self.delay.value()
        has_timer = any(a.kind == "timer" for a in board.arrows)
        deadline = self.timer_deadline
        if has_timer and deadline is None and self.timer_running.isChecked():
            deadline = self.captured_at+self.countdown.value()
            self.timer_deadline = deadline
        if has_timer and not self.timer_confirmed.isChecked():
            self.message("请先核对子弹倒计时")
            return
        flight_wait, not_before = self.flight_wait.value(), self.next_action_at
        duration = self.countdown.value() if has_timer else None
        auto_next = self.auto_next.isChecked()
        captured_at = self.captured_at
        orbit_options = dict(orbit_speed=self.orbit_speed.value(),orbit_window=self.orbit_window.value(),
                             orbit_clockwise=self.orbit_direction.currentIndex()==0)

        def run(report):
            run_dir = export_run(self.run_root, source, board, plan)
            report("run_directory", {"path":str(run_dir)})
            executor = Executor(device, board, plan, source, run_dir, control, report, delay,
                                timer_deadline=deadline, timer_duration=duration, flight_wait=flight_wait, not_before=not_before,
                                captured_at=captured_at,**orbit_options)
            cursor = executor.run(start)
            if auto_next:
                return ContinuousPlay(device,control,self.run_root,report,delay=delay,
                                      flight_wait=flight_wait,
                                      **orbit_options,
                                      next_report=lambda kind,data: executor.log(kind,**data)).run()
            return cursor
        self.work(run, self.execution_done, execution=True)

    def advance_level(self):
        if self.busy or self.editing or not self.selected_serial():
            return
        serial = self.selected_serial()
        self.control = ExecutionControl()
        device,control = AndroidDevice(serial,self.adb),self.control
        auto_next,delay,flight_wait = self.auto_next.isChecked(),self.delay.value(),self.flight_wait.value()
        orbit_options = dict(orbit_speed=self.orbit_speed.value(),orbit_window=self.orbit_window.value(),
                             orbit_clockwise=self.orbit_direction.currentIndex()==0)
        def run(report):
            if auto_next:
                return ContinuousPlay(device,control,self.run_root,report,delay=delay,
                                      flight_wait=flight_wait,**orbit_options).run()
            return NextLevel(device,control,report).run()
        self.work(run,self.execution_done,execution=True)

    def execution_event(self, kind, data):
        if kind == "auto_level_source":
            self.set_source(data['image'],self.selected_serial(),data['captured_at'])
            self.progress_label.setText("已进入下一关，正在自动识别…")
            return
        if kind == "auto_level_recognized":
            self.board = data['board']
            self.crop = self.board.crop_rect
            self.view.show_crop(self.crop)
            self.countdown.blockSignals(True)
            self.countdown.setValue(self.board.timer_seconds if self.board.timer_seconds is not None else -1)
            self.countdown.blockSignals(False)
            self.summary.setText(f"自动识别 {len(self.board.arrows)} 条箭头 · {len(self.board.turn_channels)} 个转弯通道 · {self.board.cols} 列 × {self.board.rows} 行")
            for issue in self.board.issues:
                self.message(issue)
            self.populate()
            self.redraw()
            self.update_timer_text()
            self.orbit_group.setVisible(self.board.orbit_obstacle is not None)
            return
        if kind == "auto_level_plan":
            self.plan = data['plan']
            self.populate()
            self.redraw()
            self.message(f"新关卡已通过执行前校验，自动执行 {len(self.plan.order)} 步")
            return
        if kind == "timer_started":
            self.timer_deadline = data["deadline"]
            self.update_timer_text()
        if kind == "budget":
            self.progress_label.setText(f"清障剩余 {data['steps']} 步，预计 {data['estimate']:.1f} 秒 / 倒计时 {max(0,data['remaining']):.1f} 秒")
            return
        if kind in ("attempted", "sent"):
            self.uncertain = True
        elif kind in ("dispatched", "assumed_departed"):
            self.cursor = data["step"]
            self.uncertain = False
            self.next_action_at = data.get("not_before",self.next_action_at)
            label = "已发送/预计离场（未核验）"
            self.progress_label.setText(f"{label} {self.cursor} / {data['total']} 步")
            self.list.setCurrentRow(min(self.cursor, len(self.plan.order) - 1))
        elif kind == "waiting_timer":
            self.progress_label.setText(f"等待子弹 #{data['arrow']} 到时并离场，约 {data['seconds']:.1f} 秒；可停止")
        elif kind == "waiting_counter":
            self.progress_label.setText(f"等待 {data['threshold']} 次消除的离场与方块解锁，约 {data['seconds']:.1f} 秒；可停止")
        elif kind == "waiting_channel":
            self.progress_label.setText(f"等待箭头通过转弯通道 {data['channels']}，约 {data['seconds']:.1f} 秒；可停止")
        elif kind == "waiting_orbit":
            self.progress_label.setText(f"等待小豆豆通过箭头 #{data['arrow']} 的出口，约 {data['seconds']:.1f} 秒；可停止")
        elif kind == "next_level_waiting":
            self.progress_label.setText(data['message']+'；每 5 秒检查，可停止')
        elif kind == "next_level_clicked":
            self.progress_label.setText("已点击下一关，等待新棋盘；每 5 秒检查，可停止")
        self.message(f"{kind}: {json.dumps(data, ensure_ascii=False)}")

    def execution_done(self, cursor):
        if isinstance(cursor,NextLevelResult):
            self.set_source(cursor.image,self.selected_serial(),cursor.captured_at)
            self.message("已进入下一关并更新截图，请识别并核对新棋盘")
            return
        self.cursor = cursor
        self.uncertain = False

    def stop(self):
        if self.control:
            self.control.cancel.set()
            self.message("正在停止；已发送的点击无法撤回")

    def closeEvent(self, event):
        if self.busy:
            self.closing = True
            self.stop()
            self.message("正在结束后台任务，完成后关闭窗口…")
            event.ignore()
            return
        self.mirror.close()
        event.accept()
