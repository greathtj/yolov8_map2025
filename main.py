import sys
import os
import time
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

# Worker for running the FPS calculation in a separate thread
class FPSWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    result = Signal(str)

    def __init__(self, model_path, image_folder):
        super().__init__()
        self.model_path = model_path
        self.image_folder = image_folder

    def run(self):
        try:
            model = YOLO(self.model_path)
            image_files = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                self.result.emit("No images found in the specified folder.")
                self.finished.emit()
                return

            # Warm-up run
            self.progress.emit("Warming up the model...\n")
            model(image_files[0], verbose=False)

            elapsed_times = []
            num_images = len(image_files)

            self.progress.emit("Starting FPS measurement...\n")
            for i, image_path in enumerate(image_files):
                start_time = time.time()
                model(image_path, verbose=False)
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                current_fps = 1 / elapsed_time if elapsed_time > 0 else 0
                self.progress.emit(f"Processed image {i+1}/{num_images} ({os.path.basename(image_path)}) - Time: {elapsed_time:.4f}s - FPS: {current_fps:.2f}\n")

            if num_images > 2:
                elapsed_times.sort()
                # Exclude the fastest and slowest times
                trimmed_times = elapsed_times[1:-1]
                total_time = sum(trimmed_times)
                avg_fps = (num_images - 2) / total_time if total_time > 0 else 0
                self.result.emit(f"\nFPS calculation finished.\nAverage FPS (excluding lowest and highest): {avg_fps:.2f}\n")
            else:
                total_time = sum(elapsed_times)
                avg_fps = num_images / total_time if total_time > 0 else 0
                self.result.emit(f"\nFPS calculation finished.\nAverage FPS: {avg_fps:.2f}\n")

        except Exception as e:
            self.result.emit(f"An error occurred during FPS calculation: {e}\n")
        finally:
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
        self.ui.pushButtonSelectData.clicked.connect(self.select_data_folder)
        self.ui.pushButtonRunMAP.clicked.connect(self.run_validation)
        self.ui.pushButtonRunFPS.clicked.connect(self.run_fps_calculation)

    def append_text(self, text):
        self.ui.textBrowserOutput.insertPlainText(text)
        self.ui.textBrowserOutput.ensureCursorVisible()

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Model Files (*.pt)")
        if file_path:
            self.ui.lineEditModelValidate.setText(file_path)

    def select_data_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Validation Data Folder")
        if folder_path:
            self.ui.lineEditValidationData.setText(folder_path)

    def run_validation(self):
        self.ui.textBrowserOutput.clear()
        self.ui.pushButtonRunMAP.setEnabled(False)

        self.stream_emitter = StreamEmitter()
        self.stream_emitter.textWritten.connect(self.append_text)

        # --- Validation Worker Thread ---
        model_path = self.ui.lineEditModelValidate.text()
        data_path = f"{self.ui.lineEditValidationData.text()}/data.yaml"
        
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
        self.ui.pushButtonRunMAP.setEnabled(True)
        print("Validation complete.") # This will print to the console

    def run_fps_calculation(self):
        self.ui.textBrowserOutput.clear()
        self.ui.pushButtonRunFPS.setEnabled(False)

        model_path = self.ui.lineEditModelValidate.text()
        image_folder = f"{self.ui.lineEditValidationData.text()}/val/images"

        self.fps_thread = QThread()
        self.fps_worker = FPSWorker(model_path, image_folder)
        self.fps_worker.moveToThread(self.fps_thread)
        self.fps_worker.progress.connect(self.append_text)
        self.fps_worker.result.connect(self.append_text)

        self.fps_thread.started.connect(self.fps_worker.run)
        self.fps_worker.finished.connect(self.on_fps_finished)
        self.fps_worker.finished.connect(self.fps_thread.quit)
        self.fps_worker.finished.connect(self.fps_worker.deleteLater)
        self.fps_thread.finished.connect(self.fps_thread.deleteLater)
        
        self.fps_thread.start()

    def on_fps_finished(self):
        self.ui.pushButtonRunFPS.setEnabled(True)
        self.append_text("FPS calculation process complete.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())
