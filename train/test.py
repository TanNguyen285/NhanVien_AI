import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QLabel, QWidget)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer
from PIL import Image
from torchvision import transforms
from collections import deque

# Import kiến trúc mô hình từ file model.py của bạn
from test.model import PAttLite

class EmotionCameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. Cấu hình Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PAttLite(num_classes=8)
        
        model_path = 'PAtt_Lite_V2_PyTorch_Final.pth'
        #model_path = 'checkpoints/best_phase_2.pth'
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded!")
        except Exception as e:
            print(f"Lỗi load model: {e}")

        self.class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 
                            'Happy', 'Neutral', 'Sad', 'Surprise']

        # --- CẤU HÌNH SMOOTHING ---
        # Lưu xác suất của 10 frame gần nhất
        self.history_size = 10 
        self.prob_history = deque(maxlen=self.history_size)
        # --------------------------

        # 2. Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.capture = cv2.VideoCapture(0)
        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle('PAttLite V2 - Real-time Smoothing')
        self.setGeometry(100, 100, 800, 750)

        layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: black; border: 3px solid #333;")
        layout.addWidget(self.video_label)

        self.result_label = QLabel('Đang phân tích...')
        self.result_label.setFont(QFont('Arial', 22, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("color: #00FF00; background-color: #222; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.result_label)

        self.btn_quit = QPushButton('Thoát')
        self.btn_quit.setFixedHeight(40)
        self.btn_quit.clicked.connect(self.close)
        layout.addWidget(self.btn_quit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret: return

        frame = cv2.flip(frame, 1)

        # Dự đoán với Smoothing
        label, confidence = self.predict_emotion_smooth(frame)

        # Vẽ lên màn hình
        color = (0, 255, 0) if label != 'Neutral' else (255, 255, 255)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        self.result_label.setText(f"{label} - {confidence:.1f}%")

        # Chuyển đổi hiển thị PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(720, 480, Qt.AspectRatioMode.KeepAspectRatio))

    def predict_emotion_smooth(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Lấy xác suất (Softmax) thay vì lấy luôn argmax
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

        # Thêm xác suất hiện tại vào hàng đợi history
        self.prob_history.append(probabilities)

        # Tính trung bình cộng xác suất của các frame trong history
        avg_probabilities = np.mean(self.prob_history, axis=0)
        
        # Lấy nhãn có xác suất trung bình cao nhất
        pred_idx = np.argmax(avg_probabilities)
        confidence = avg_probabilities[pred_idx] * 100

        return self.class_names[pred_idx], confidence

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EmotionCameraGUI()
    gui.show()
    sys.exit(app.exec())