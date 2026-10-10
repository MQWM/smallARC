from PySide6.QtCore import QThread, Signal


class Task(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    event = Signal(str, object)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.succeeded.emit(self.function(self.event.emit))
        except Exception as exc:
            self.failed.emit(str(exc))

