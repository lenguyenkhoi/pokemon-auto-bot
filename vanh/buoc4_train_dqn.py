"""
BƯỚC 4 — Train DQN (board 8x6, ~10-30 phút)
=============================================
ĐÂY LÀ PHẦN LÕI CHÍNH của TV3.

DQN khác Q-Learning ở chỗ:
  - Dùng MẠNG NEURAL thay cho bảng tra cứu
  - Học QUY LUẬT CHUNG, áp dụng được cho bàn cờ mới chưa từng thấy
  - Hai kỹ thuật ổn định: Experience Replay + Target Network

Lệnh chạy:
    python buoc4_train_dqn.py

Kết quả mong đợi:
  - Loss (sai số mạng neural) GIẢM dần hoặc ổn định
  - Reward trung bình có xu hướng TĂNG
  - File dqn_model.pt được lưu → bàn giao cho TV5/TV6
  - File dqn_results.png được lưu → dùng trong báo cáo
"""

import torch
print("=" * 55)
print("BƯỚC 4: Train DQN trên board 8x6")
print("=" * 55)
print(f"Thiết bị dùng để train: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

print("""
DQN hoạt động thế nào?
─────────────────────────────────────────────
1. AI chụp "ảnh" bàn cờ (ma trận số) + 1 action cụ thể
   → encode thành 1 vector dài, đưa vào mạng neural
   → mạng trả ra 1 số = Q-value (ước tính "độ tốt" của action đó)

2. Với mỗi action hợp lệ, tính Q-value → chọn action cao nhất.

3. Sau mỗi bước đi, lưu (state, action, reward, state_mới) vào bộ nhớ đệm.

4. Lấy ngẫu nhiên 64 kinh nghiệm từ bộ nhớ → train mạng bằng gradient descent.
   (Đây là Experience Replay: tránh học lệch do dữ liệu liên tiếp tương quan nhau)

5. Cứ 200 bước, copy trọng số sang Target Network.
   (Giúp "mục tiêu" ổn định, không bị mạng tự đuổi chính mình)

Cột "Loss" khi chạy: sai số giữa Q-value mạng dự đoán và Q-value đúng thật.
  Loss GIẢM = mạng đang học đúng hướng.
  Loss dao động nhẹ = bình thường, RL vốn không ổn định như supervised learning.
─────────────────────────────────────────────
""")

n_ep = int(input("Nhập số episode muốn train (khuyến nghị 1500, tối thiểu 500): ") or "1500")
print(f"\n>>> Bắt đầu train {n_ep} ván trên board 8x6, {8} loại Pokémon...")
print(">>> In tiến độ mỗi 50 ván. Hãy theo dõi cột Loss và Tỉ lệ thắng.\n")

from dqn_agent import train_dqn, plot_dqn_results
import numpy as np

agent, rewards, wins, losses = train_dqn(
    n_episodes=n_ep,
    rows=8, cols=6,
    n_types=8,
    batch_size=64,
)

# ─── LƯU KẾT QUẢ ─────────────────────────────────────────
print("\n>>> Vẽ biểu đồ và lưu model...")
plot_dqn_results(rewards, wins, losses, save_path="dqn_results.png")
agent.save("dqn_model.pt")

# ─── PHÂN TÍCH KẾT QUẢ ───────────────────────────────────
print("\n" + "=" * 55)
print("PHÂN TÍCH KẾT QUẢ")
print("=" * 55)

# Loss: so sánh 10% đầu vs 10% cuối
if losses:
    chunk = max(1, len(losses) // 10)
    first_loss = np.mean(losses[:chunk])
    last_loss  = np.mean(losses[-chunk:])
    print(f"\nLoss trung bình 10% ĐẦU : {first_loss:.3f}")
    print(f"Loss trung bình 10% CUỐI: {last_loss:.3f}")
    if last_loss < first_loss * 0.8:
        print("→ ✓ Loss GIẢM RÕ — mạng đang học tốt.")
    elif last_loss < first_loss:
        print("→ ✓ Loss có xu hướng giảm — mạng đang học, cần thêm episode để thấy rõ hơn.")
    else:
        print("→ ⚠ Loss chưa giảm rõ. Xem ghi chú debug bên dưới.")

# Reward: so sánh 20% đầu vs 20% cuối
chunk_ep = max(1, len(rewards) // 5)
first_reward = np.mean(rewards[:chunk_ep])
last_reward  = np.mean(rewards[-chunk_ep:])
print(f"\nReward TB 20% ĐẦU : {first_reward:.1f}")
print(f"Reward TB 20% CUỐI: {last_reward:.1f}")
if last_reward > first_reward + 10:
    print("→ ✓ Reward TĂNG — AI đang học cách nối nhiều cặp hơn mỗi ván.")
elif last_reward > first_reward:
    print("→ ✓ Reward có xu hướng tăng nhẹ.")
else:
    print("→ ⚠ Reward chưa tăng rõ. Có thể cần train thêm episode.")

# Tỉ lệ thắng
overall_win = np.mean(wins) * 100
last_win    = np.mean(wins[-min(200, len(wins)):]) * 100
print(f"\nTỉ lệ thắng toàn bộ : {overall_win:.1f}%")
print(f"Tỉ lệ thắng 200 ván cuối: {last_win:.1f}%")
if last_win > overall_win:
    print("→ ✓ Tỉ lệ thắng TĂNG dần theo thời gian — AI đang tiến bộ.")

print(f"""
─────────────────────────────────────────────
Ghi chú: Nếu tỉ lệ thắng vẫn thấp/0%:
  - Validator giả lập (chỉ nối thẳng hàng/cột + chữ L) còn hạn chế.
  - Khi TV4 xong, ghép validator thật → bàn cờ ít bế tắc hơn → tỉ lệ thắng tăng hẳn.
  - Cần train thêm episode (thử 3000-5000).
  - Reward tăng dần và Loss giảm là dấu hiệu AI đang học đúng hướng,
    dù tỉ lệ thắng chưa cao vì validator còn giả lập.
─────────────────────────────────────────────
""")

print("=" * 55)
print("✓ Bước 4 hoàn thành! Đã lưu:")
print("  - dqn_model.pt   ← file model bàn giao TV6")
print("  - dqn_results.png ← biểu đồ dùng trong báo cáo")
print("\nTiếp tục sang Bước 5 để kiểm tra model.")
print("  Lệnh: python buoc5_kiem_tra_model.py")
print("=" * 55)
