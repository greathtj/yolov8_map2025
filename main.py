import sys
import os
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QFont # Import QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from main_ui import Ui_MainWindow
import qdarkstyle
from ultralytics import YOLO

class StreamEmitter(QObject):
    textWritten = Signal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass # No-op

# Worker for running the validation in a separate thread
class ValidationWorker(QObject):
    finished = Signal()

    def __init__(self, model_path, data_path, stream_emitter):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        self.stream_emitter = stream_emitter

    def run(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = self.stream_emitter
            sys.stderr = self.stream_emitter

            model = YOLO(self.model_path)
            metrics = model.val(data=self.data_path)
            print(f"\nValidation finished.")
            print(f"mAP50-95: {metrics.box.map}")  # mAP50-95
            print(f"mAP50: {metrics.box.map50}")  # mAP50
            print(f"mAP75: {metrics.box.map75}")  # mAP75
            print(f"list of mAP50-95 for each category: {metrics.box.maps}")  # list of mAP50-95 for each category

        except Exception as e:
            print(f"An error occurred during validation: {e}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.finished.emit()

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Set monospaced font for textBrowserOutput
        font = self.ui.textBrowserOutput.font()
        font.setFamily("Monospace") # Or "Courier New", "Consolas, 'Fira Code', 'Hack', 'Cascadia Code', etc."
        font.setPointSize(10) # Adjust size if needed
        self.ui.textBrowserOutput.setFont(font)

        self.ui.pushButtonExit.clicked.connect(QApplication.instance().quit)
        self.ui.pushButtonSelectModel.clicked.connect(self.select_model_file)
        self.ui.pushButtonSelectData.clicked.connect(self.select_data_file)
        self.ui.pushButtonRunValidation.clicked.connect(self.run_validation)

    def append_text(self, text):
        self.ui.textBrowserOutput.insertPlainText(text)
        self.ui.textBrowserOutput.ensureCursorVisible()

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Model Files (*.pt)")
        if file_path:
            self.ui.lineEditModelValidate.setText(file_path)

    def select_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.ui.lineEditValidationData.setText(file_path)

    def run_validation(self):
        self.ui.textBrowserOutput.clear()
        self.ui.pushButtonRunValidation.setEnabled(False)

        self.stream_emitter = StreamEmitter()
        self.stream_emitter.textWritten.connect(self.append_text)

        # --- Validation Worker Thread ---
        model_path = self.ui.lineEditModelValidate.text()
        data_path = self.ui.lineEditValidationData.text()
        
        self.validation_thread = QThread()
        self.validation_worker = ValidationWorker(model_path, data_path, self.stream_emitter)
        self.validation_worker.moveToThread(self.validation_thread)

        self.validation_thread.started.connect(self.validation_worker.run)
        self.validation_worker.finished.connect(self.on_validation_finished)
        self.validation_worker.finished.connect(self.validation_thread.quit)
        self.validation_worker.finished.connect(self.validation_worker.deleteLater)
        self.validation_thread.finished.connect(self.validation_thread.deleteLater)
        
        self.validation_thread.start()
        # ----------------------------

    def on_validation_finished(self):
        self.ui.pushButtonRunValidation.setEnabled(True)
        print("Validation complete.") # This will print to the console

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())
