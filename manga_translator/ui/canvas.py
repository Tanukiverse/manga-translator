"""Page canvas: shows one layer, editable region boxes, and a brush for the text mask."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

MODE_SELECT, MODE_ADD, MODE_PAINT, MODE_ERASE = "select", "add", "paint", "erase"
HANDLE = 10


def to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        h, w = img.shape
        return QImage(np.ascontiguousarray(img).data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class RegionItem(QGraphicsRectItem):
    def __init__(self, canvas: "Canvas", rid: int, bbox):
        x, y, w, h = bbox
        super().__init__(0, 0, w, h)
        self.canvas, self.rid = canvas, rid
        self.setPos(x, y)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.label = QGraphicsSimpleTextItem(str(rid), self)
        f = QFont()
        f.setPixelSize(18)
        f.setBold(True)
        self.label.setFont(f)
        self.label.setBrush(QColor("#d81b60"))
        self.label.setPos(2, -22)
        self._resizing = False
        self._press_state = None
        self.set_style(False)

    def set_style(self, locked: bool):
        color = QColor("#1e88e5") if not locked else QColor("#43a047")
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(28)
        self.setBrush(QBrush(fill))

    def paint(self, painter, option, widget=None):
        if self.isSelected():
            pen = QPen(QColor("#ff9800"), 3)
            pen.setCosmetic(True)
            self.setPen(pen)
        super().paint(painter, option, widget)
        r = self.rect()
        painter.fillRect(QRectF(r.right() - HANDLE, r.bottom() - HANDLE, HANDLE, HANDLE), QColor("#ff9800"))

    def _on_handle(self, pos: QPointF) -> bool:
        r = self.rect()
        return pos.x() >= r.right() - HANDLE * 1.5 and pos.y() >= r.bottom() - HANDLE * 1.5

    def hoverMoveEvent(self, e):
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if self._on_handle(e.pos()) else Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, e):
        self._press_state = self.bbox()
        self._resizing = e.button() == Qt.MouseButton.LeftButton and self._on_handle(e.pos())
        super().mousePressEvent(e)
        self.canvas.region_selected.emit(self.rid)

    def mouseMoveEvent(self, e):
        if self._resizing:
            p = e.pos()
            self.prepareGeometryChange()
            self.setRect(0, 0, max(12, p.x()), max(12, p.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self._resizing = False
        if self._press_state is not None and self._press_state != self.bbox():
            self.canvas.region_changed.emit(self.rid, self.bbox())
        self._press_state = None

    def bbox(self) -> list[int]:
        r = self.rect()
        return [int(round(self.pos().x())), int(round(self.pos().y())), int(round(r.width())), int(round(r.height()))]


class Canvas(QGraphicsView):
    region_selected = Signal(int)
    region_changed = Signal(int, list)  # id, bbox
    region_added = Signal(list)  # bbox
    mask_edit_started = Signal()
    mask_edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#2b2b2b"))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.image_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setOpacity(0.45)
        self.scene().addItem(self.image_item)
        self.scene().addItem(self.mask_item)
        self.items_by_id: dict[int, RegionItem] = {}
        self.mode = MODE_SELECT
        self.brush = 14
        self.mask: np.ndarray | None = None
        self.show_mask = False
        self.show_boxes = True
        self._drag_start: QPointF | None = None
        self._rubber: QGraphicsRectItem | None = None
        self._last_paint: QPointF | None = None

    # ---- content -----------------------------------------------------
    def set_image(self, img: np.ndarray | None, fit: bool = False):
        if img is None:
            self.image_item.setPixmap(QPixmap())
            return
        self.image_item.setPixmap(QPixmap.fromImage(to_qimage(img)))
        self.scene().setSceneRect(QRectF(0, 0, img.shape[1], img.shape[0]))
        if fit:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_mask(self, mask: np.ndarray | None):
        self.mask = mask
        self._refresh_mask()

    def _refresh_mask(self):
        if self.mask is None or not self.show_mask:
            self.mask_item.setVisible(False)
            return
        h, w = self.mask.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = self.mask
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self.mask_item.setPixmap(QPixmap.fromImage(qimg))
        self.mask_item.setVisible(True)

    def set_show_mask(self, on: bool):
        self.show_mask = on
        self._refresh_mask()

    def set_regions(self, regions, selected: int | None = None):
        for item in self.items_by_id.values():
            self.scene().removeItem(item)
        self.items_by_id.clear()
        for r in regions:
            item = RegionItem(self, r.id, r.bbox)
            item.set_style(r.locked)
            item.setVisible(self.show_boxes)
            self.scene().addItem(item)
            self.items_by_id[r.id] = item
        if selected in self.items_by_id:
            self.items_by_id[selected].setSelected(True)

    def set_show_boxes(self, on: bool):
        self.show_boxes = on
        for item in self.items_by_id.values():
            item.setVisible(on)

    def select_region(self, rid: int):
        self.scene().clearSelection()
        item = self.items_by_id.get(rid)
        if item:
            item.setSelected(True)
            self.ensureVisible(item, 40, 40)

    def set_mode(self, mode: str):
        self.mode = mode
        movable = mode == MODE_SELECT
        for item in self.items_by_id.values():
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, movable)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if mode != MODE_SELECT else Qt.CursorShape.ArrowCursor)

    # ---- zoom --------------------------------------------------------
    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit(self):
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ---- mouse -------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            return super().mousePressEvent(e)
        p = self.mapToScene(e.position().toPoint())
        if self.mode == MODE_ADD and e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = p
            self._rubber = self.scene().addRect(QRectF(p, p), QPen(QColor("#ff9800"), 2, Qt.PenStyle.DashLine))
            return
        if self.mode in (MODE_PAINT, MODE_ERASE) and self.mask is not None and e.button() == Qt.MouseButton.LeftButton:
            self.mask_edit_started.emit()
            self._last_paint = p
            self._paint(p, p)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        if self._rubber is not None and self._drag_start is not None:
            self._rubber.setRect(QRectF(self._drag_start, p).normalized())
            return
        if self._last_paint is not None:
            self._paint(self._last_paint, p)
            self._last_paint = p
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            super().mouseReleaseEvent(e)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        if self._rubber is not None:
            r = self._rubber.rect()
            self.scene().removeItem(self._rubber)
            self._rubber, self._drag_start = None, None
            if r.width() > 8 and r.height() > 8:
                self.region_added.emit([int(r.x()), int(r.y()), int(r.width()), int(r.height())])
            return
        if self._last_paint is not None:
            self._last_paint = None
            self.mask_edited.emit()
            return
        super().mouseReleaseEvent(e)

    def _paint(self, a: QPointF, b: QPointF):
        value = 255 if self.mode == MODE_PAINT else 0
        cv2.line(self.mask, (int(a.x()), int(a.y())), (int(b.x()), int(b.y())), value, self.brush * 2, cv2.LINE_8)
        self._refresh_mask()
