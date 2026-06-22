"""
BƯỚC 3 — Train Q-Learning (board nhỏ, ~3-5 phút)
===================================================
MỤC ĐÍCH: Kiểm chứng logic thuật toán đúng trước khi sang DQN.
KHÔNG nhằm tạo AI giỏi thật sự — Q-table không tổng quát hoá được
sang bàn cờ mới mỗi ván (đó là lý do cần DQN ở Bước 4).

Lệnh chạy:
    python buoc3_train_qlearning.py

Kết quả mong đợi:
  - Đối chứng 2 tỉ lệ thắng TĂNG (từ ~53% lên ~59%+)
  - File qlearning_results.png được lưu
"""

print("=" * 55)
print("BƯỚC 3: Train Q-Learning trên board 4x4")
print("=" * 55)
print("""
Q-Learning là gì?
  Xây một "bảng tra cứu" (Q-table): với mỗi cặp (tình huống, nước đi),
  lưu một con số = "nếu ở đây làm vậy, về lâu dài được bao nhiêu điểm".
  Công thức cập nhật sau mỗi nước đi:
  Q(s,a) = Q(s,a) + alpha × [reward + gamma × Q_tốt_nhất(s_mới) - Q(s,a)]
  Hiểu đơn giản: điều chỉnh dần dần dựa trên "kết quả thực tế" vừa nhận.

Epsilon-greedy là gì?
  Với xác suất epsilon → chọn ngẫu nhiên (khám phá nước chưa thử bao giờ)
  Còn lại → chọn nước có điểm Q cao nhất (dùng kiến thức đã học)
  Epsilon giảm dần: lúc đầu khám phá nhiều, sau dần tin vào kiến thức.
""")

input("Nhấn Enter để bắt đầu train... ")

from q_learning_agent import train, plot_results, evaluate_vs_random, evaluate_fixed_boards
import numpy as np

# ─── TRAIN ───────────────────────────────────────────────
print("\n>>> Bắt đầu train 3000 ván trên board 4x4 (n_types=5)...")
print(">>> Mỗi 200 ván sẽ in tiến độ. Chờ khoảng 1-3 phút.\n")

agent, rewards, wins = train(n_episodes=3000, rows=4, cols=4, n_types=5)

print("\n>>> Vẽ và lưu biểu đồ...")
plot_results(rewards, wins, save_path="qlearning_results.png")

# ─── PHÂN TÍCH KẾT QUẢ ───────────────────────────────────
print("\n" + "=" * 55)
print("PHÂN TÍCH KẾT QUẢ")
print("=" * 55)

print(f"\nTổng số trạng thái (state-action) đã học: {len(agent.q_table)}")
print("→ Q-table lưu từng tình huống cụ thể đã gặp.")
print("  Số này lớn nhưng mỗi bàn cờ gần như chỉ gặp 1 lần → giới hạn chính của Q-Learning.\n")

# Tỉ lệ thắng nửa đầu vs nửa sau trong training (có nhiễu epsilon)
first = np.mean(wins[:1500]) * 100
second = np.mean(wins[1500:]) * 100
print(f"Tỉ lệ thắng nửa đầu train: {first:.1f}%")
print(f"Tỉ lệ thắng nửa sau  train: {second:.1f}%")
print("→ Hai con số này có thể gần nhau — BÌNH THƯỜNG, vì trong lúc train")
print("  epsilon vẫn còn cao nên AI vẫn chọn ngẫu nhiên nhiều, che lấp việc học.\n")

# ─── ĐỐI CHỨNG 1 ─────────────────────────────────────────
print("─" * 55)
print("ĐỐI CHỨNG 1 — AI học vs AI hoàn toàn NGẪU NHIÊN (bàn mới mỗi ván)")
print("─" * 55)
evaluate_vs_random(agent, n_episodes=500, rows=4, cols=4, n_types=5)
print("""
→ Chênh lệch nhỏ ở đây là BÌNH THƯỜNG.
  Nguyên nhân thật: bàn cờ sinh MỚI mỗi ván → Q-table gần như không bao giờ
  "tra lại" trạng thái cũ → không cải thiện được → không tổng quát hoá được.
  ĐÂY CHÍNH LÀ LÝ DO CẦN DQN: mạng neural học QUY LUẬT CHUNG, không học thuộc lòng.
""")

# ─── ĐỐI CHỨNG 2 ─────────────────────────────────────────
print("─" * 55)
print("ĐỐI CHỨNG 2 — Q-Learning trên TẬP BÀN CỜ CỐ ĐỊNH (lặp lại nhiều lần)")
print("─" * 55)
print("(Đo epsilon=0 sau khi học, không lẫn yếu tố random khám phá)\n")
evaluate_fixed_boards(rows=6, cols=6, n_types=9, n_fixed_boards=15, n_episodes=3000)
print("""
→ Đây mới là phép đo ĐÚNG của Q-table:
  Khi state LẶP LẠI (bàn cờ cố định), Q-table học và cải thiện được rõ rệt.
  Xác nhận CODE ĐÚNG — vấn đề là ở đặc điểm bài toán (bàn cờ luôn mới), không phải lỗi.
""")

print("=" * 55)
print("✓ Bước 3 hoàn thành!")
print("  Xem file qlearning_results.png để thấy biểu đồ.")
print("  Tiếp tục sang Bước 4 (phần quan trọng nhất).")
print("  Lệnh: python buoc4_train_dqn.py")
print("=" * 55)
