import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from custom import EmotionCNN

class EmotionEngine:
    def __init__(self, yolo_path, emotion_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load YOLO Detect mặt
        self.detector = YOLO(yolo_path)
        
        # 2. Load CNN
        self.classifier = EmotionCNN(num_classes=6)
        try:
            self.classifier.load_state_dict(torch.load(emotion_path, map_location=self.device))
            self.classifier.to(self.device).eval()
        except Exception as e:
            print(f"Lỗi load trọng số CNN: {e}")

        # 3. Cấu hình xử lý ảnh
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['Angry', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def predict_frame(self, frame):
        """Nhận vào 1 frame BGR, trả về (frame_da_ve, label_chu)"""
        h, w, _ = frame.shape
        
        # --- QUAN TRỌNG: Khởi tạo mặc định để không bị lỗi 'referenced before assignment' ---
        current_final_label = "No Face" 
        
        # Chạy YOLO detect mặt
        results = self.detector.predict(frame, imgsz=320, conf=0.5, verbose=False)

        # Duyệt qua từng khuôn mặt phát hiện được
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                ox1, oy1, ox2, oy2 = box.astype(int)
                
                # Cắt mặt + Padding 15% để lấy đủ trán/tai giúp AI đoán chuẩn hơn
                mw, mh = int((ox2-ox1)*0.15), int((oy2-oy1)*0.15)
                x1, y1 = max(0, ox1-mw), max(0, oy1-mh)
                x2, y2 = min(w, ox2+mw), min(h, oy2+mh)
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0: continue

                try:
                    # Tiền xử lý ảnh mặt
                    face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

                    # Predict cảm xúc
                    with torch.no_grad():
                        output = self.classifier(face_tensor)
                        prob = torch.softmax(output, dim=1)
                        conf, pred = torch.max(prob, 1)
                        
                        # Lấy nhãn và độ tự tin
                        label_idx = pred.item()
                        current_final_label = self.labels[label_idx]
                        conf_val = conf.item() * 100

                    # Vẽ lên frame
                    color = (0, 255, 0) # Màu xanh lá
                    label_str = f"{current_final_label} {conf_val:.1f}%"
                    
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
                    cv2.putText(frame, label_str, (ox1, oy1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception as e:
                    print(f"Lỗi xử lý vùng mặt: {e}")
                    continue

        # Trả về frame đã vẽ và nhãn của mặt cuối cùng phát hiện được
        return frame, current_final_label