from pathlib import Path
import time

from PySide6.QtWidgets import QApplication, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
import pytest

from arrow_solver.ui.main_window import MainWindow
from arrow_solver.vision import read_image, recognize
from arrow_solver.next_level import NextLevelResult


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path):
    window = MainWindow(tmp_path)
    yield window
    window.close()


def sample():
    return read_image(Path(__file__).parent / "fixtures" / "level2.jpg")


def test_moving_orbit_preview_reuses_scene_items_and_hides_for_normal_level(window,app,monkeypatch):
    from arrow_solver.orbit import detect,measure,predicted_position
    fixtures = Path(__file__).parent/'fixtures/recorded'
    first = detect(read_image(fixtures/'level332.png'))
    image = read_image(fixtures/'level332-later.png')
    second = measure(first,detect(image),.968007)
    monkeypatch.setattr('arrow_solver.ui.main_window.time.monotonic',lambda:100.)
    window.set_source(image,'fake',99.,second)
    assert window.orbit_speed.value() == pytest.approx(second.measured_speed,abs=.01)
    circle,path,text = window.view.orbit_items
    center = circle.rect().center()
    expected = predicted_position(second,1,window.orbit_speed.value(),True)
    assert (center.x(),center.y()) == pytest.approx(expected)
    original_items = len(window.view.scene().items())
    monkeypatch.setattr('arrow_solver.ui.main_window.time.monotonic',lambda:101.)
    window.update_orbit_preview()
    assert circle.rect().center()!=center
    assert len(window.view.scene().items())==original_items
    assert '预测位置' in text.text() and path.path().elementCount()>1
    window.set_source(sample(),'fake',101.)
    assert not any(item.isVisible() for item in window.view.orbit_items)


def test_preview_never_enables_offline_execution(window, app):
    image = sample()
    window.set_source(image, None)
    window.recognized(recognize(image))
    window.device_box.addItem("fake device", "fake")
    window.reviewed.setChecked(True)
    window.list.setCurrentRow(15)
    assert window.cursor == 0
    assert window.preview == 15
    assert not window.run_button.isEnabled()
    assert window.list.count() == 59


def test_next_level_updates_source_and_invalidates_previous_plan(window,app,monkeypatch):
    image = sample()
    window.set_source(image,'fake')
    window.recognized(recognize(image))
    window.device_box.addItem('fake','fake')
    window.reviewed.setChecked(True)
    window.cursor = 10
    assert not window.auto_next.isChecked()
    monkeypatch.setattr('arrow_solver.ui.main_window.NextLevel.run',
                        lambda self:NextLevelResult(image.copy(),123.))
    window.advance_level()
    assert window.busy
    assert window.stop_button.isEnabled()
    deadline = time.monotonic()+5
    while window.busy and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    assert not window.busy
    assert window.board is None and window.plan is None
    assert window.cursor == 0
    assert window.captured_at == 123.
    assert window.source_device == 'fake'
    assert not window.reviewed.isChecked()
    assert not window.run_button.isEnabled()
    assert window.recognize_button.isEnabled()


def test_auto_next_dispatches_continuous_runner_and_updates_level_state(window,app,monkeypatch):
    from arrow_solver.solver import solve
    from test_executor import scene
    from arrow_solver.executor import Cancelled
    image,board = scene()
    plan = solve(board)
    window.device_box.addItem('fake','fake')
    window.auto_next.setChecked(True)
    def run(runner):
        runner.report('auto_level_source',{'image':image,'captured_at':234.})
        runner.report('auto_level_recognized',{'board':board})
        runner.report('auto_level_plan',{'plan':plan})
        runner.report('dispatched',{'step':1,'total':2,'not_before':235.})
        raise Cancelled('test stop')
    monkeypatch.setattr('arrow_solver.ui.main_window.ContinuousPlay.run',run)
    window.advance_level()
    deadline = time.monotonic()+5
    while window.busy and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    assert not window.busy
    assert window.board is board and window.plan is plan
    assert window.cursor == 1
    assert window.source_device == 'fake'
    assert window.captured_at == 234.
    assert window.timer_deadline is None
    assert not window.reviewed.isChecked()
    assert not window.run_button.isEnabled()


def test_execution_requires_bound_device_and_review(window, app):
    image = sample()
    window.set_source(image, "phone-a")
    window.recognized(recognize(image))
    window.device_box.addItem("phone-a", "phone-a")
    assert not window.run_button.isEnabled()
    window.reviewed.setChecked(True)
    assert window.run_button.isEnabled()
    window.device_box.addItem("phone-b", "phone-b")
    window.device_box.setCurrentIndex(1)
    assert not window.run_button.isEnabled()


def test_worker_finishes_and_updates_ui(window, app):
    window.set_source(sample(), None)
    window.analyze()
    assert window.busy
    assert not window.recognize_button.isEnabled()
    deadline = time.monotonic() + 10
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert not window.busy
    assert window.plan is not None
    assert len(window.plan.order) == 59


def test_timer_first_tap_and_explicit_confirmation(window, app):
    image = read_image(Path(__file__).parent / "fixtures/recorded/level201.png")
    captured = time.monotonic()-20
    window.set_source(image, "fake", captured)
    window.recognized(recognize(image))
    window.device_box.addItem("fake", "fake")
    assert not window.reviewed.isChecked()
    assert not window.busy
    assert window.cursor == 0
    assert window.countdown.value() == 45
    window.reviewed.setChecked(True)
    assert not window.run_button.isEnabled()
    window.timer_confirmed.setChecked(True)
    assert window.run_button.isEnabled()
    window.update_timer_text()
    assert window.captured_at == captured
    assert window.timer_deadline is None
    assert "尚未启动" in window.timer_note.text()
    window.timer_running.setChecked(True)
    assert not window.timer_confirmed.isChecked()
    assert "已启动" in window.timer_note.text()
    window.timer_running.setChecked(False)
    window.timer_confirmed.setChecked(True)
    window.execution_event("timer_started", {"deadline": time.monotonic()+45})
    anchored = window.timer_deadline
    window.update_timer_text()
    assert window.timer_deadline == anchored
    assert window.run_button.isEnabled()
    assert window.captured_at == captured
    window.execution_event("dispatched", {"step":1,"total":74,"not_before":time.monotonic()+.4})
    window._update_controls()
    assert "未核验" in window.progress_label.text()


def test_crop_is_visible_without_manual_selection(window,app):
    window.set_source(sample(),None)
    assert window.view.crop_outline.isVisible()
    assert window.view.crop_outline.rect().getRect() == tuple(map(float,window.crop))
    window.recognized(recognize(sample()))
    assert window.view.crop_outline.rect().getRect() == tuple(map(float,window.board.crop_rect))


def test_removed_execution_controls_and_log_follow_latest(window,app):
    labels = [button.text() for button in window.findChildren(QPushButton)]
    assert not {'导入截图','网格参数','执行一步并核验','暂停'} & set(labels)
    assert not hasattr(window,'mode_box')
    window.show()
    for i in range(600):
        window.message(f'第 {i} 条日志')
    app.processEvents()
    assert window.log.document().blockCount() <= 500
    bar = window.log.verticalScrollBar()
    assert bar.maximum()>0
    bar.setValue(0)
    window.message('最新日志')
    app.processEvents()
    assert bar.value() == bar.maximum()


def setup_editor(window,app):
    from test_executor import scene
    from types import SimpleNamespace
    image,board = scene()
    window.set_source(image,None)
    window.recognized(SimpleNamespace(board=board))
    window.show()
    app.processEvents()


def click_image(window,point):
    QTest.mouseClick(window.view.viewport(),Qt.MouseButton.LeftButton,
                     pos=window.view.mapFromScene(*point))


def test_click_to_draw_arrow_save_and_invalidate_review(window,app):
    setup_editor(window,app)
    window.reviewed.setChecked(True)
    window.edit_arrow(new=True)
    assert not window.run_button.isEnabled()
    click_image(window,(20,100))
    click_image(window,(100,100))
    assert len(window.view.edit_points)==2
    QTest.keyClick(window.view,Qt.Key.Key_Return)
    assert not window.editing
    arrow = window.board.arrows[-1]
    assert arrow.cells==[(0,2),(1,2),(2,2)]
    assert arrow.direction=='RIGHT'
    assert arrow.click_point==(100,100)
    assert window.plan is not None
    assert not window.reviewed.isChecked()


def test_click_edit_selects_arrow_and_cancel_preserves_board(window,app):
    setup_editor(window,app)
    fingerprint = window.board.fingerprint()
    window.edit_arrow()
    click_image(window,(80,60))
    assert window.edit_target.id==1
    window.pick_arrow_click()
    click_image(window,(96,60))
    assert abs(window.edit_click[0]-96)<1
    QTest.keyClick(window.view,Qt.Key.Key_Escape)
    assert not window.editing
    assert window.board.fingerprint()==fingerprint


def test_drag_control_point_and_undo_without_typing(window,app):
    setup_editor(window,app)
    window.point_selected((80,60))
    window.edit_arrow()
    first = window.view.mapFromScene(*window.view.edit_points[0])
    target = window.view.mapFromScene(60,100)
    QTest.mousePress(window.view.viewport(),Qt.MouseButton.LeftButton,pos=first)
    QTest.mouseMove(window.view.viewport(),target)
    QTest.mouseRelease(window.view.viewport(),Qt.MouseButton.LeftButton,pos=target)
    assert window.view.edit_points[0] == (60,100)
    QTest.mouseClick(window.view.viewport(),Qt.MouseButton.RightButton,pos=target)
    assert len(window.view.edit_points)==1
    window.save_arrow_edit()
    assert window.editing
    assert '至少' in window.edit_note.text()


def test_orbit_settings_only_appear_for_moving_obstacle_board(window,app):
    image=read_image(Path(__file__).parent/'fixtures/recorded/level332.png')
    window.set_source(image,'fake',time.monotonic())
    window.recognized(recognize(image))
    assert not window.orbit_group.isHidden()
    assert window.orbit_speed.value()==118
    assert window.orbit_window.value()==3
    window.set_source(sample(),'fake')
    window.recognized(recognize(sample()))
    assert window.orbit_group.isHidden()
