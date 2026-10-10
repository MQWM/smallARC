import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QEvent
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QPainterPath, QColor, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem


class BoardView(QGraphicsView):
    crop_selected = Signal(object)
    point_selected = Signal(object)
    point_hovered = Signal(object)
    path_changed = Signal(object)
    edit_finished = Signal()
    edit_cancelled = Signal()
    click_point_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.pixmap_item = None
        self.crop_mode = False
        self.start = None
        self.rectangle = None
        self.crop_outline = None
        self.edit_points = None
        self.edit_items = []
        self.drag_vertex = None
        self.pick_click = False
        self.snap_point = lambda point: point
        self.edit_click_point = None
        self.auto_fit = True
        self.orbit_items = None
        self.setMinimumSize(420, 540)

    def set_image(self, bgr, reset=False):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        if self.pixmap_item is None:
            self.pixmap_item = self.scene().addPixmap(pixmap)
            reset = True
        else:
            self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        if reset:
            self.fit_image()
            self.point_hovered.emit(None)

    def fit_image(self):
        if self.pixmap_item:
            self.auto_fit = True
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def show_orbit(self, center, radius=0., trail=(), caption=''):
        if center is None:
            for item in self.orbit_items or ():
                item.hide()
            return
        if self.orbit_items is None:
            pen = QPen(QColor(0,230,255),2)
            pen.setCosmetic(True)
            circle = self.scene().addEllipse(QRectF(),pen,QBrush(QColor(0,230,255,100)))
            path = self.scene().addPath(QPainterPath(),pen)
            text = self.scene().addSimpleText('')
            text.setBrush(QBrush(QColor(0,230,255)))
            text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            self.orbit_items = circle,path,text
            for item in self.orbit_items:
                item.setZValue(4)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        circle,path,text = self.orbit_items
        x,y = center
        circle.setRect(QRectF(x-radius,y-radius,radius*2,radius*2))
        shape = QPainterPath()
        if len(trail)>1:
            shape.moveTo(*trail[0])
            for p in trail[1:]:
                shape.lineTo(*p)
            last,previous = np.array(trail[-1]),np.array(trail[-2])
            direction = last-previous
            length = np.linalg.norm(direction)
            if length>0:
                direction /= length
                normal = np.array([-direction[1],direction[0]])
                tip_size = 12/max(.05,abs(self.transform().m11()))
                for side in (-1,1):
                    shape.moveTo(*last)
                    shape.lineTo(*(last-direction*tip_size+normal*side*tip_size*.5))
        path.setPath(shape)
        text.setText(caption)
        anchor = self.mapFromScene(QPointF(x,y))
        label_x = max(4,min(anchor.x()+radius*abs(self.transform().m11())+8,
                          self.viewport().width()-text.boundingRect().width()-4))
        label_y = max(4,min(anchor.y()-radius*abs(self.transform().m11())-25,
                          self.viewport().height()-text.boundingRect().height()-4))
        text.setPos(self.mapToScene(round(label_x),round(label_y)))
        for item in self.orbit_items:
            item.show()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.pixmap_item and self.auto_fit:
            self.fitInView(self.pixmap_item,Qt.AspectRatioMode.KeepAspectRatio)

    def begin_crop(self):
        self.crop_mode = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def show_crop(self, rect):
        if self.crop_outline is None:
            self.crop_outline = self.scene().addRect(QRectF(),QPen(Qt.GlobalColor.yellow,2))
            self.crop_outline.setZValue(2)
            pen = self.crop_outline.pen()
            pen.setCosmetic(True)
            self.crop_outline.setPen(pen)
        self.crop_outline.setRect(QRectF(*rect))
        self.crop_outline.show()

    def begin_path(self, points, click_point=None):
        self.crop_mode = False
        self.edit_points = [tuple(p) for p in points]
        self.edit_click_point = click_point
        self.auto_fit = False
        if self.edit_points:
            points = np.array(self.edit_points)
            left,top = points.min(axis=0)-40
            width,height = np.ptp(points,axis=0)+80
            self.fitInView(QRectF(left,top,width,height),Qt.AspectRatioMode.KeepAspectRatio)
        elif self.crop_outline:
            self.fitInView(self.crop_outline,Qt.AspectRatioMode.KeepAspectRatio)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.draw_path()

    def end_path(self):
        self.edit_points = None
        self.drag_vertex = None
        self.pick_click = False
        self.edit_click_point = None
        self.draw_path()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.unsetCursor()

    def draw_path(self):
        for item in self.edit_items:
            self.scene().removeItem(item)
        self.edit_items = []
        if not self.edit_points:
            return
        path = QPainterPath()
        path.moveTo(*self.edit_points[0])
        for point in self.edit_points[1:]:
            path.lineTo(*point)
        pen = QPen(Qt.GlobalColor.cyan,3)
        pen.setCosmetic(True)
        self.edit_items.append(self.scene().addPath(path,pen))
        radius = 5/max(.05,self.transform().m11())
        for index,(x,y) in enumerate(self.edit_points):
            item = self.scene().addEllipse(x-radius,y-radius,radius*2,radius*2,pen)
            self.edit_items.append(item)
            label = self.scene().addSimpleText(str(index+1))
            label.setBrush(Qt.GlobalColor.cyan)
            label.setScale(1/max(.05,self.transform().m11()))
            label.setPos(x+radius,y-radius)
            self.edit_items.append(label)
        for item in self.edit_items:
            item.setZValue(3)
        if self.edit_click_point is not None:
            x,y = self.edit_click_point
            click_pen = QPen(Qt.GlobalColor.red,2)
            click_pen.setCosmetic(True)
            self.edit_items.append(self.scene().addEllipse(x-radius*1.4,y-radius*1.4,radius*2.8,radius*2.8,click_pen))
            self.edit_items[-1].setZValue(4)

    def undo_vertex(self):
        if self.edit_points:
            self.edit_points.pop()
            self.draw_path()
            self.path_changed.emit(list(self.edit_points))

    def keyPressEvent(self,event):
        if self.edit_points is not None:
            if event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
                self.edit_finished.emit()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.edit_cancelled.emit()
                return
            if event.key() in (Qt.Key.Key_Backspace,Qt.Key.Key_Delete):
                self.undo_vertex()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        self.auto_fit = False
        self.scale(1.15 if event.angleDelta().y() > 0 else 1 / 1.15,
                   1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        self.show_hover_point(event.position().toPoint())
        if self.edit_points is not None:
            self.draw_path()

    def show_hover_point(self, viewport_point):
        point = self.mapToScene(viewport_point)
        rect = self.sceneRect()
        if self.pixmap_item and 0 <= point.x() < rect.width() and 0 <= point.y() < rect.height():
            self.point_hovered.emit((int(np.floor(point.x())), int(np.floor(point.y()))))
        else:
            self.point_hovered.emit(None)

    def viewportEvent(self, event):
        if event.type() == QEvent.Type.Leave:
            self.point_hovered.emit(None)
        return super().viewportEvent(event)

    def mousePressEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        if self.edit_points is not None:
            if event.button() == Qt.MouseButton.RightButton:
                self.undo_vertex()
                return
            if event.button() == Qt.MouseButton.LeftButton and self.sceneRect().contains(point):
                p = (point.x(),point.y())
                if self.pick_click:
                    self.pick_click = False
                    self.edit_click_point = p
                    self.draw_path()
                    self.click_point_changed.emit(p)
                    return
                for i,vertex in enumerate(self.edit_points):
                    screen = self.mapFromScene(*vertex)
                    if np.hypot(screen.x()-event.position().x(),screen.y()-event.position().y()) <= 6:
                        self.drag_vertex = i
                        return
                self.edit_points.append(self.snap_point(p))
                self.draw_path()
                self.path_changed.emit(list(self.edit_points))
            return
        if self.crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self.start = point
            self.rectangle = QGraphicsRectItem(QRectF(point, point))
            self.rectangle.setPen(QPen(Qt.GlobalColor.yellow, 2))
            self.scene().addItem(self.rectangle)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.point_selected.emit((point.x(), point.y()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.edit_points is not None and self.drag_vertex is not None:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.edit_points[self.drag_vertex] = self.snap_point((point.x(),point.y()))
                self.draw_path()
                self.path_changed.emit(list(self.edit_points))
            self.show_hover_point(event.position().toPoint())
            return
        if self.crop_mode and self.start is not None:
            self.rectangle.setRect(QRectF(self.start, self.mapToScene(event.position().toPoint())).normalized())
            self.show_hover_point(event.position().toPoint())
            return
        super().mouseMoveEvent(event)
        self.show_hover_point(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if self.edit_points is not None:
            self.drag_vertex = None
            return
        if self.crop_mode and self.start is not None:
            rect = self.rectangle.rect().intersected(self.sceneRect())
            self.scene().removeItem(self.rectangle)
            self.rectangle = None
            self.start = None
            self.crop_mode = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
            if rect.width() > 10 and rect.height() > 10:
                self.crop_selected.emit((int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())))
            return
        super().mouseReleaseEvent(event)
