# TV3 — RL Engineer
## Xây dựng AI chơi Pokémon Nối Hình (Onet)

---

## Cài đặt (làm 1 lần duy nhất)

```bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt (Windows)
venv\Scripts\activate
# Kích hoạt (Mac/Linux)
source venv/bin/activate

# 3. Cài thư viện
pip install torch numpy matplotlib
```

---

## Chạy theo thứ tự từng bước

| Bước | Lệnh | Mục đích | Thời gian |
|------|------|----------|-----------|
| 1 | `python buoc1_kiem_tra_moi_truong.py` | Xác nhận Python + thư viện cài đúng | < 10 giây |
| 2 | `python buoc2_test_moi_truong_game.py` | Xem bàn cờ giả lập, thử nối ô | < 10 giây |
| 3 | `python buoc3_train_qlearning.py` | Train Q-Learning, xác nhận logic đúng | ~3-5 phút |
| 4 | `python buoc4_train_dqn.py` | **Train DQN — phần lõi chính** | ~10-30 phút |
| 5 | `python buoc5_kiem_tra_model.py` | Kiểm tra model, demo predict | < 1 phút |

---

## Cấu trúc file

```
tv3_final/
├── buoc1_kiem_tra_moi_truong.py   ← Chạy đầu tiên
├── buoc2_test_moi_truong_game.py
├── buoc3_train_qlearning.py
├── buoc4_train_dqn.py              ← Phần lõi chính
├── buoc5_kiem_tra_model.py
│
├── onet_env.py          ← Bàn cờ giả lập (môi trường AI tập chơi)
├── q_learning_agent.py  ← Q-Learning (baseline, xác nhận logic)
├── dqn_agent.py         ← DQN (AI thật sự, dùng mạng neural)
│
├── dqn_model.pt         ← [SINH RA sau Bước 4] Model bàn giao TV6
├── dqn_results.png      ← [SINH RA sau Bước 4] Biểu đồ DQN
└── qlearning_results.png← [SINH RA sau Bước 3] Biểu đồ Q-Learning
```

---

## Cách TV5/TV6 dùng model (sau khi bạn bàn giao dqn_model.pt)

```python
from dqn_agent import DQNAgent

# Load model 1 lần khi khởi động
agent = DQNAgent.load("dqn_model.pt")

# Trong vòng lặp chơi game:
action = agent.predict(board, valid_actions)
# board         ← numpy array từ TV1 (CV đọc màn hình)
# valid_actions ← list cặp ô từ TV4 (path_validator)
# action        → ((r1,c1), (r2,c2)) để TV5 click chuột
# action = None → bàn cờ bế tắc, TV5 cần bấm nút Xáo bài
```

---

## Khi TV4 xong — Ghép validator thật

Mở `onet_env.py`, tìm lúc khởi tạo `OnetEnv` và sửa:

```python
# Thêm import ở đầu file
from search.path_validator import find_valid_path   # module của TV4

# Khi tạo env, truyền validator thật vào
env = OnetEnv(rows=8, cols=6, n_types=8, validator=find_valid_path)
```

Sau đó chạy lại Bước 4 để train với luật nối đúng thật.

---

## Biểu đồ kết quả — dùng trong báo cáo

**Q-Learning (qlearning_results.png):**
Chứng minh logic đúng bằng Đối chứng 2:
tỉ lệ thắng tăng từ ~53% → ~59% khi dùng tập bàn cờ cố định.

**DQN (dqn_results.png):**
- Loss giảm → mạng đang học đúng hướng
- Reward tăng → AI chọn nước đi tốt hơn theo thời gian
- Tỉ lệ thắng → cần validator thật của TV4 để thấy rõ nhất

---

## Kết quả mong đợi cuối cùng (sau khi ghép validator thật)

- AI tự chơi được 1 ván hoàn chỉnh mà không cần can thiệp tay
- Tỉ lệ thắng cao hơn AI random baseline (Đối chứng 1)
- Predict dưới 100ms mỗi lần gọi (đủ nhanh cho pipeline real-time)
