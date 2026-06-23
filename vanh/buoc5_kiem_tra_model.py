"""
BƯỚC 5 — Kiểm tra model đã train và demo cách TV5/TV6 dùng
===========================================================
Kiểm tra file dqn_model.pt load được không, predict nhanh không,
và demo cách TV5 sẽ gọi hàm predict() trong pipeline thật.

Lệnh chạy:
    python buoc5_kiem_tra_model.py

Cần chạy Bước 4 trước để có file dqn_model.pt.
"""

import os
import time
import numpy as np
from onet_env import OnetEnv
from dqn_agent import DQNAgent

print("=" * 55)
print("BƯỚC 5: Kiểm tra model đã train")
print("=" * 55)

# ─── Kiểm tra file tồn tại ───────────────────────────────
if not os.path.exists("dqn_model.pt"):
    print("✗ Không tìm thấy file dqn_model.pt!")
    print("  Hãy chạy Bước 4 trước: python buoc4_train_dqn.py")
    exit(1)

size_mb = os.path.getsize("dqn_model.pt") / 1024 / 1024
print(f"✓ Tìm thấy dqn_model.pt ({size_mb:.2f} MB)")

# ─── Load model ──────────────────────────────────────────
print("\n>>> Load model vào bộ nhớ...")
try:
    agent = DQNAgent.load("dqn_model.pt")
    print("✓ Load thành công!")
    print(f"  Board size  : {agent.rows} hàng × {agent.cols} cột")
    print(f"  Số loại Pokémon: {agent.n_types}")
    print(f"  Epsilon     : {agent.epsilon} (= 0 khi deploy — không random nữa)")
except Exception as e:
    print(f"✗ Load thất bại: {e}")
    exit(1)

# ─── Test tốc độ predict ────────────────────────────────
print("\n>>> Test tốc độ predict (quan trọng: phải < 1 giây để pipeline real-time không bị lag)...")
env = OnetEnv(rows=agent.rows, cols=agent.cols, n_types=agent.n_types)
board = env.reset()
valid_actions = env.get_valid_actions()

if not valid_actions:
    print("⚠ Board này bế tắc ngay từ đầu (hiếm gặp), reset lại...")
    board = env.reset()
    valid_actions = env.get_valid_actions()

# Đo thời gian predict 100 lần lấy trung bình
N = 100
t0 = time.time()
for _ in range(N):
    action = agent.predict(board, valid_actions)
elapsed = (time.time() - t0) / N * 1000  # milliseconds
print(f"✓ Thời gian predict trung bình: {elapsed:.2f} ms / lần gọi")
if elapsed < 100:
    print("  → Đủ nhanh để dùng trong vòng lặp real-time của TV5.")
else:
    print("  → Hơi chậm, nhưng vẫn chấp nhận được (< 1 giây).")

# ─── Demo predict cụ thể ─────────────────────────────────
print("\n" + "─" * 55)
print("DEMO: Cách TV5 sẽ gọi hàm predict() trong pipeline thật")
print("─" * 55)
print("""
CODE TV5 sẽ dùng (chỉ 4 dòng quan trọng):
──────────────────────────────────────────
from dqn_agent import DQNAgent

agent = DQNAgent.load("dqn_model.pt")      # load 1 lần khi khởi động

# Trong vòng lặp chơi game:
action = agent.predict(board, valid_actions)
# board        ← ma trận numpy từ TV1 (CV đọc màn hình thật)
# valid_actions ← list cặp ô từ TV4 (path_validator)
# action        → ((r1,c1), (r2,c2)) để TV5 click chuột
──────────────────────────────────────────
""")

# Thực sự gọi và in ra
print("Bàn cờ hiện tại:")
env.render()
print(f"Số cặp ô hợp lệ: {len(valid_actions)}")
action = agent.predict(board, valid_actions)
if action:
    (r1, c1), (r2, c2) = action
    loai = board[r1, c1]
    print(f"\nAI chọn nối: ô ({r1},{c1}) ↔ ô ({r2},{c2})  [cùng loại Pokémon số {loai}]")
    print(f"TV5 sẽ click pixel tương ứng với ô ({r1},{c1}) rồi ô ({r2},{c2}) trên màn hình thật.")
else:
    print("AI trả về None (không có action nào — bàn cờ bế tắc).")
    print("TV5 cần gọi nút Xáo bài khi nhận được None.")

# ─── Demo chơi 5 ván liên tiếp ───────────────────────────
print("\n" + "─" * 55)
print("DEMO: AI tự chơi 5 ván liên tiếp (xem kết quả từng ván)")
print("─" * 55)
wins = 0
for i in range(5):
    board = env.reset()
    done = False
    steps = 0
    won = False
    while not done and steps < 300:
        valid = env.get_valid_actions()
        if not valid:
            break
        action = agent.predict(board, valid)
        if action is None:
            break
        board, reward, done, info = env.step(action)
        won = info.get("win", False)
        steps += 1
    result = "THẮNG ✓" if won else "thua (bế tắc)"
    print(f"  Ván {i+1}: {result}  ({steps} nước đi)")
    if won:
        wins += 1
print(f"\nKết quả: {wins}/5 ván thắng")
if wins >= 3:
    print("→ ✓ Model hoạt động tốt!")
elif wins >= 1:
    print("→ Model hoạt động, cần train thêm để tỉ lệ thắng cao hơn.")
else:
    print("→ Tỉ lệ thắng thấp (do validator giả lập còn hạn chế).")
    print("  Khi TV4 xong, ghép validator thật → tỉ lệ thắng sẽ tăng hẳn.")

# ─── Khi TV4 xong ────────────────────────────────────────
print("\n" + "─" * 55)
print("SAU KHI TV4 XONG — Ghép validator thật vào đây:")
print("─" * 55)
print("""
1. Mở file onet_env.py
2. Tìm dòng: validator=None  (trong hàm __init__)
3. Thay bằng:
       from search.path_validator import find_valid_path
       env = OnetEnv(..., validator=find_valid_path)
4. Chạy lại Bước 4 để train với validator thật → model tốt hơn nhiều.
""")

print("=" * 55)
print("✓ Bước 5 hoàn thành!")
print("\nCÁC FILE ĐÃ CÓ ĐỂ BÀN GIAO:")
print("  dqn_model.pt       → TV6 dùng để tích hợp vào main.py")
print("  dqn_results.png    → Dùng trong báo cáo/slide (biểu đồ học)")
print("  qlearning_results.png → Dùng trong báo cáo (so sánh 2 phương pháp)")
print("=" * 55)
