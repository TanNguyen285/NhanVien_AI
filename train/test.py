import os
import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. KHU VỰC TÙY CHỈNH
# =====================================================================
DATASET_PATH = r"C:\Users\ThisPC\Desktop\data_emo\emooo\data\train" 
OUTPUT_H5_FILE = "emodata_112x112_gray.h5"

TARGET_SIZE = 112  # <-- Đã đổi thành 112
CHANNELS = 1       # <-- Ảnh xám (Grayscale)

# Tỷ lệ chia tập dữ liệu
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
# =====================================================================

def process_merged_dataset():
    print(f"Đang kiểm tra thư mục: {DATASET_PATH} ...\n")
    
    if not os.path.exists(DATASET_PATH):
        print(f" LỖI: Không tìm thấy đường dẫn {DATASET_PATH}")
        return

    # Lấy danh sách các class và lọc đúng 7 class
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    # Kiểm tra nếu thừa/thiếu class
    if len(class_names) != 7:
        print(f" CẢNH BÁO: Tìm thấy {len(class_names)} thư mục, nhưng bạn yêu cầu 7 class.")
        print(f" Danh sách hiện có: {class_names}")

    all_images = []
    all_labels = []

    print("=== BẮT ĐẦU ĐỌC DỮ LIỆU (GRAYSCALE 112x112) ===")
    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_PATH, class_name)
        image_files = os.listdir(class_path)
        
        count_per_class = 0
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            
            # 1. Đọc ảnh Grayscale trực tiếp
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            # 2. Resize về 112x112
            image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
            
            # 3. Thêm chiều channel (112, 112) -> (112, 112, 1)
            image = np.expand_dims(image, axis=-1)
            
            all_images.append(image)
            all_labels.append(class_index)
            count_per_class += 1
            
        print(f"  - Thư mục '{class_name}': {count_per_class} ảnh.")

    X = np.array(all_images, dtype='uint8') # Lưu uint8 để file h5 nhẹ hơn
    y = np.array(all_labels)
    print(f"\n -> Tổng cộng đọc được: {len(X)} ảnh.")
    print(f" -> Kích thước mảng X: {X.shape}") # Sẽ có dạng (N, 112, 112, 1)

    if len(X) == 0:
        print("LỖI: Không đọc được ảnh nào!")
        return

    # 3. Chia dữ liệu
    print(f"\nĐang chia dữ liệu...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_SIZE), random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  + Tập Train: {len(X_train)} ảnh")
    print(f"  + Tập Valid: {len(X_val)} ảnh")
    print(f"  + Tập Test:  {len(X_test)} ảnh")

    # 4. Lưu ra file h5
    print(f"\nĐang đóng gói và lưu vào file {OUTPUT_H5_FILE}...")
    with h5py.File(OUTPUT_H5_FILE, 'w') as hf:
        hf.create_dataset('X_train', data=X_train, compression="gzip")
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('X_val', data=X_val, compression="gzip")
        hf.create_dataset('y_val', data=y_val)
        hf.create_dataset('X_test', data=X_test, compression="gzip")
        hf.create_dataset('y_test', data=y_test)
        
    print(f"--- HOÀN TẤT! File lưu tại: {os.path.abspath(OUTPUT_H5_FILE)} ---")

if __name__ == "__main__":
    process_merged_dataset()