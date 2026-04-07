import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ptflops import get_model_complexity_info
from model import MicroViT 

class MicroViTEvaluator:
    def __init__(self, model_variant='S2', num_classes=8, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Khởi tạo model
        self.model = MicroViT(variant=model_variant, num_classes=num_classes)
        
        # 2. Load trọng số thực tế sau khi train
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[INFO] Đã nạp thành công trọng số từ: {checkpoint_path}")
        else:
            print("[WARNING] Không tìm thấy checkpoint, đang đánh giá model với trọng số ngẫu nhiên!")
            
        self.model.to(self.device)
        self.model.eval()

    def profile_complexity(self):
        """ Đo Params và FLOPs (Giống Table II trong bài báo) """
        print("\n--- Đang phân tích Complexity (FLOPs/Params) ---")
        macs, params = get_model_complexity_info(
            self.model, (3, 224, 224), 
            as_strings=True, 
            print_per_layer_stat=False, 
            verbose=False
        )
        return {"FLOPs": macs, "Parameters": params}

    def profile_speed(self, batch_size=1, iterations=100):
        """ Đo Latency và FPS (Giống Table IV trong bài báo) """
        print(f"--- Đang đo tốc độ thực tế (Batch Size = {batch_size}) ---")
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # Warm-up (Khởi động GPU)
        with torch.no_grad():
            for _ in range(20):
                _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)
                
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        total_time = time.time() - start_time
        fps = (iterations * batch_size) / total_time
        latency = (total_time / iterations) * 1000 # ms
        
        return {"Latency (ms)": f"{latency:.2f}", "Throughput (FPS)": f"{fps:.2f}"}

    def load_and_plot_history(self, history_path='training_history.pth'):
        """ Đọc file history từ quá trình train và vẽ biểu đồ """
        if not os.path.exists(history_path):
            print(f"[ERROR] Không tìm thấy file {history_path}. Hãy chạy train.py trước!")
            return

        history_data = torch.load(history_path)
        sns.set_theme(style="whitegrid")
        
        # Tương thích với các tên key khác nhau
        losses = history_data.get('train_loss', [])
        accs = history_data.get('val_acc', [])
        epochs = range(1, len(losses) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Vẽ đường Loss (Trục trái)
        ax1.set_xlabel('Epochs (Số vòng lặp)')
        ax1.set_ylabel('Loss (Độ mất mát)', color='tab:red')
        ax1.plot(epochs, losses, color='tab:red', marker='o', linewidth=2, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Vẽ đường Accuracy (Trục phải)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%) (Độ chính xác)', color='tab:blue')
        ax2.plot(epochs, accs, color='tab:blue', marker='s', linewidth=2, label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        plt.title('MicroViT Performance Report (AffectNet)', fontsize=16)
        fig.tight_layout()
        plt.savefig('final_evaluation_report.png')
        print("[SUCCESS] Đã xuất biểu đồ báo cáo: final_evaluation_report.png")
        plt.show()

# ==========================================
# CHƯƠNG TRÌNH CHẠY ĐÁNH GIÁ
# ==========================================
if __name__ == "__main__":
    # 1. Khởi tạo (Sửa num_classes cho đúng với bộ dữ liệu của bạn, AffectNet thường là 7 hoặc 8)
    # Trỏ checkpoint_path đến file tốt nhất bạn đã train được
    tester = MicroViTEvaluator(
        model_variant='S2', 
        num_classes=8, 
        checkpoint_path='checkpoints/best_microvit.pth'
    )

    # 2. Thực hiện các phép đo
    complexity = tester.profile_complexity()
    speed = tester.profile_speed(batch_size=1) # Đo Batch=1 để xem tốc độ trên thiết bị Edge

    # 3. Hiển thị bảng tổng hợp
    report = {**complexity, **speed}
    print("\n" + "="*60)
    print(" BÁO CÁO CHI TIẾT MÔ HÌNH MICROVIT")
    print("="*60)
    print(pd.DataFrame([report]).to_string(index=False))
    print("="*60)

    # 4. Vẽ biểu đồ từ dữ liệu train thực tế
    tester.load_and_plot_history('training_history.pth')