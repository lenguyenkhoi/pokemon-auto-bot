"""
BƯỚC 1 — Kiểm tra môi trường
==============================
Chạy file này TRƯỚC TIÊN để chắc mọi thứ cài đúng.

Lệnh chạy:
    python buoc1_kiem_tra_moi_truong.py
"""

print("=" * 55)
print("BƯỚC 1: Kiểm tra môi trường cài đặt")
print("=" * 55)

# Kiểm tra Python version
import sys
print(f"\n✓ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("✗ Cần Python 3.8 trở lên! Hãy cài lại Python.")
    sys.exit(1)

# Kiểm tra từng thư viện
libs = {
    "numpy": "numpy",
    "torch": "torch",
    "matplotlib": "matplotlib",
}

all_ok = True
for name, module in libs.items():
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"✓ {name} version {ver} — OK")
    except ImportError:
        print(f"✗ {name} CHƯA CÀI — chạy: pip install {name}")
        all_ok = False

if not all_ok:
    print("\n>>> Cài thiếu thư viện. Chạy lệnh sau rồi thử lại:")
    print("    pip install torch numpy matplotlib")
    sys.exit(1)

# Kiểm tra GPU (không bắt buộc nhưng hữu ích để biết)
import torch
if torch.cuda.is_available():
    print(f"\n✓ GPU khả dụng: {torch.cuda.get_device_name(0)}")
    print("  → DQN sẽ train NHANH HƠN nhiều nhờ GPU.")
else:
    print("\n⚠ Không tìm thấy GPU — sẽ dùng CPU.")
    print("  → DQN train chậm hơn (~10-30 phút), nhưng vẫn chạy được bình thường.")

print("\n" + "=" * 55)
print("✓ Môi trường đã sẵn sàng! Tiếp tục sang Bước 2.")
print("  Lệnh: python buoc2_test_moi_truong_game.py")
print("=" * 55)
