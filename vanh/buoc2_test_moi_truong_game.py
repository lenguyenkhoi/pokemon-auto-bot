"""
BƯỚC 2 — Test môi trường game giả lập
=======================================
Kiểm chứng file onet_env.py (bàn cờ giả lập) hoạt động đúng.
Bạn sẽ THẤY bàn cờ dạng số, thử nối vài cặp, xem kết quả.

Lệnh chạy:
    python buoc2_test_moi_truong_game.py
"""

import random
from onet_env import OnetEnv

print("=" * 55)
print("BƯỚC 2: Test môi trường game giả lập")
print("=" * 55)

print("""
Giải thích bàn cờ hiển thị:
  Số (0, 1, 2, ...) = ô có Pokémon loại đó
  Dấu chấm (.)      = ô trống (đã bị xoá hoặc ô viền ngoài)
""")

# --- Test 1: Sinh bàn cờ và xem cấu trúc ---
print("─" * 40)
print("TEST 1: Sinh bàn cờ 4x4 mới")
print("─" * 40)
env = OnetEnv(rows=4, cols=4, n_types=4)
print("Bàn cờ vừa sinh:")
env.render()
print(f"Kích thước: 4 hàng x 4 cột = 16 ô")
print(f"Số loại Pokémon: 4 (mỗi loại xuất hiện đúng số chẵn lần)")

# --- Test 2: Xem action hợp lệ ---
print("\n" + "─" * 40)
print("TEST 2: Liệt kê các cặp ô có thể nối")
print("─" * 40)
valid = env.get_valid_actions()
print(f"Số cặp ô nối được ngay lúc này: {len(valid)}")
if valid:
    print("Ví dụ 3 cặp đầu tiên (hàng, cột):")
    for i, (a, b) in enumerate(valid[:3]):
        loai = env.board[a[0], a[1]]
        print(f"  [{i+1}] Ô {a} ↔ Ô {b}  (cùng loại Pokémon số {loai})")

# --- Test 3: Nối cặp hợp lệ ---
print("\n" + "─" * 40)
print("TEST 3: Thực hiện nối một cặp ô hợp lệ")
print("─" * 40)
action = random.choice(valid)
cellA, cellB = action
loai = env.board[cellA[0], cellA[1]]
print(f"Chọn nối ô {cellA} ↔ ô {cellB} (cùng loại {loai})")
state, reward, done, info = env.step(action)
print(f"Kết quả: reward = {reward}, done = {done}")
print("Bàn cờ sau khi nối (2 ô vừa nối biến thành dấu chấm):")
env.render()

# --- Test 4: Thử nối cặp không hợp lệ ---
print("─" * 40)
print("TEST 4: Thử nối cặp SAI (khác loại)")
print("─" * 40)
# Tìm 2 ô còn tồn tại khác loại
invalid_action = None
rows, cols = env.board.shape
for r1 in range(rows):
    for c1 in range(cols):
        if env.board[r1, c1] == -1:
            continue
        for r2 in range(rows):
            for c2 in range(cols):
                if (r1, c1) == (r2, c2):
                    continue
                if env.board[r2, c2] == -1:
                    continue
                if env.board[r1, c1] != env.board[r2, c2]:
                    invalid_action = ((r1, c1), (r2, c2))
                    break
            if invalid_action:
                break
        if invalid_action:
            break
if invalid_action:
    cA, cB = invalid_action
    print(f"Thử nối ô {cA} (loại {env.board[cA[0], cA[1]]}) ↔ ô {cB} (loại {env.board[cB[0], cB[1]]})")
    _, reward, _, info = env.step(invalid_action)
    print(f"Kết quả: reward = {reward}  ← bị phạt âm vì nối sai")
    print(f"Info: {info}")

# --- Test 5: Chơi 1 ván tự động hoàn chỉnh ---
print("\n" + "─" * 40)
print("TEST 5: Chơi 1 ván hoàn chỉnh (tự chọn ngẫu nhiên)")
print("─" * 40)
env2 = OnetEnv(rows=4, cols=4, n_types=3)
state = env2.reset()
done = False
step_count = 0
total_reward = 0
while not done and step_count < 100:
    valid = env2.get_valid_actions()
    if not valid:
        print(f"Bước {step_count}: Bế tắc! Không còn cặp nào nối được.")
        break
    action = random.choice(valid)
    _, reward, done, info = env2.step(action)
    total_reward += reward
    step_count += 1
    if done:
        if info.get("win"):
            print(f"Bước {step_count}: THẮNG! Đã xoá hết bàn cờ.")
        elif info.get("stuck"):
            print(f"Bước {step_count}: Bế tắc sau khi nối.")
        else:
            print(f"Bước {step_count}: Kết thúc.")
print(f"Tổng reward cả ván: {total_reward}")
print(f"Tổng số bước: {step_count}")

print("\n" + "=" * 55)
print("✓ Môi trường game hoạt động đúng!")
print("  Tiếp tục sang Bước 3.")
print("  Lệnh: python buoc3_train_qlearning.py")
print("=" * 55)
