import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import os
from collections import deque
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Import PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QWidget, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Import kiến trúc mô hình PAttLite của bạn
try:
    from model import PAttLite
except ImportError:
    print("Lỗi: Không tìm thấy file model.py!")

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
    

class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_logs = []
        self.current_emotion = "N/A"
        self.current_confidence = 0.0

        # --- CONFIG ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_path = 'runs/detect/yolov26_e20/weights/best.pt'
        #self.emotion_path = 'runs/PAttLite_emodata.pth' # Đường dẫn file model 112x112 vừa train
        self.emotion_path = 'runs/PAtt_Test.pth' # File .h5 bạn đã train thành công

        # Đảm bảo thứ tự nhãn đúng với lúc train (7 nhãn)
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.prob_history = deque(maxlen=10)
        
        # TRANSFORM MỚI: Phải có Grayscale(3) và Resize(112)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3), # Chuyển vùng mặt cắt thành 3 kênh Gray
            transforms.Resize((112, 112)),               # Khớp size 112x112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_models()
        self.initUI()

        self.cam_thread = CameraThread()
        self.cam_thread.change_pixmap_signal.connect(self.update_frame_logic)
        self.cam_thread.start()

    def load_models(self):
        try:
            self.face_detector = YOLO(self.yolo_path)
            self.emotion_model = PAttLite(num_classes=7)
            self.emotion_model.load_state_dict(torch.load(self.emotion_path, map_location=self.device))
            self.emotion_model.to(self.device)
            self.emotion_model.eval()
            print("AI Models Loaded: Grayscale 112x112 Mode")
            print("Model:", next(self.emotion_model.parameters()).device)

        except Exception as e:
            print(f"Lỗi nạp mô hình: {e}")

    def initUI(self):
        self.setWindowTitle("AI Check-in System - Grayscale 112x112")
        self.setFixedSize(1200, 800)
        self.setStyleSheet(self.get_style())

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        self.cam_lbl = QLabel("ĐANG MỞ CAMERA...")
        self.cam_lbl.setFixedSize(720, 540)
        self.cam_lbl.setObjectName("CamBox")
        self.cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.emotion_display = QLabel("Cảm xúc: Chờ nhận diện...")
        self.emotion_display.setObjectName("EmotionStatus")
        self.emotion_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_layout = QHBoxLayout()
        self.btn_in = QPushButton("KHÁCH VÀO (IN)")
        self.btn_in.setObjectName("BtnIn")
        self.btn_in.clicked.connect(lambda: self.log_entry("CHECK-IN"))

        self.btn_out = QPushButton("KHÁCH RA (OUT)")
        self.btn_out.setObjectName("BtnOut")
        self.btn_out.clicked.connect(lambda: self.log_entry("CHECK-OUT"))

        btn_layout.addWidget(self.btn_in)
        btn_layout.addWidget(self.btn_out)

        left_layout.addWidget(self.cam_lbl)
        left_layout.addWidget(self.emotion_display)
        left_layout.addLayout(btn_layout)

        right_layout = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Giờ", "Hành động", "Cảm xúc AI"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.btn_export = QPushButton("XUẤT EXCEL & RESET")
        self.btn_export.setObjectName("BtnExport")
        self.btn_export.clicked.connect(self.export_to_excel)

        right_layout.addWidget(QLabel("NHẬT KÝ HỆ THỐNG"))
        right_layout.addWidget(self.table)
        right_layout.addWidget(self.btn_export)

        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_frame_logic(self, frame):
        frame = cv2.flip(frame, 1) 
        display_frame = frame.copy() # Giữ bản gốc RGB để hiển thị

        if self.face_detector:
            results = self.face_detector(frame, verbose=False, conf=0.5)[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Cắt vùng mặt
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Chuyển sang ảnh xám chỉ cho vùng nhận diện
                    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    
                    # Dự đoán cảm xúc (Logic transform sẽ lo việc resize 112 và channel 3)
                    label, conf = self.predict_emotion_smooth(gray_face)
                    self.current_emotion = label
                    self.current_confidence = conf

                    # Vẽ lên bản frame hiển thị (vẫn là màu để đẹp mắt)
                    color = (0, 255, 0) if label == 'Happy' else (0, 165, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{label} {conf:.0f}%", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            self.emotion_display.setText(f"AI Detection: {self.current_emotion} ({self.current_confidence:.1f}%)")

        # Hiển thị RGB
        rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.cam_lbl.setPixmap(QPixmap.fromImage(qt_img).scaled(720, 540, Qt.AspectRatioMode.KeepAspectRatio))

    def predict_emotion_smooth(self, gray_face):
        """Xử lý trên vùng mặt xám"""
        # img_tensor sẽ qua Grayscale(3) và Resize(112) trong self.transform
        img_tensor = self.transform(gray_face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.emotion_model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        self.prob_history.append(probs)
        avg_probs = np.mean(self.prob_history, axis=0)
        
        idx = np.argmax(avg_probs)
        return self.class_names[idx], avg_probs[idx] * 100

    def log_entry(self, action):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        emotion_str = f"{self.current_emotion} ({self.current_confidence:.1f}%)"
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(now))
        self.table.setItem(row, 1, QTableWidgetItem(action))
        self.table.setItem(row, 2, QTableWidgetItem(emotion_str))
        self.table.scrollToBottom()
        self.data_logs.append({"Giờ": now, "Hành động": action, "Cảm xúc": emotion_str})

    def export_to_excel(self):
        if not self.data_logs:
            QMessageBox.warning(self, "Thông báo", "Chưa có dữ liệu!")
            return
        df = pd.DataFrame(self.data_logs)
        filename = f"Emotion_Log_{datetime.datetime.now().strftime('%d%m_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        QMessageBox.information(self, "Thành công", f"Đã lưu: {filename}")
        self.data_logs = []
        self.table.setRowCount(0)

    def get_style(self):
        return """
            QMainWindow { background-color: #1a1b26; }
            QLabel { color: #a9b1d6; font-weight: bold; font-size: 15px; }
            #CamBox { border: 3px solid #7aa2f7; border-radius: 15px; background: #000; }
            #EmotionStatus { color: #bb9af7; font-size: 22px; margin: 10px; }
            QPushButton { border-radius: 10px; font-weight: bold; min-height: 55px; font-size: 14px; }
            #BtnIn { background-color: #9ece6a; color: #1a1b26; }
            #BtnOut { background-color: #f7768e; color: #1a1b26; }
            #BtnExport { background-color: #7aa2f7; color: #1a1b26; }
            QTableWidget { background-color: #24283b; color: white; border-radius: 8px; }
            QHeaderView::section { background-color: #414868; color: white; border: none; }
        """

    def closeEvent(self, event):
        self.cam_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EmotionApp()
    win.show()
    sys.exit(app.exec())