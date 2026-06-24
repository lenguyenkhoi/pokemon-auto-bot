# 🎮 Pokemon Auto Bot — AI Agent tự động chơi Pikachu Matching Game

> **Đồ án AI:** Xây dựng AI Agent dùng Deep Reinforcement Learning để tự động chơi game Pikachu Matching (Pokemon Matching) mà không cần sự can thiệp của con người.

---

## 📌 Mô tả dự án

Dự án này kết hợp hai thành phần chính:
1. **Game Engine** — Game Pikachu Matching viết bằng Python + Pygame (đã có sẵn).
2. **AI Agent** — Agent dùng Deep RL (DQN / PPO / A2C) quan sát board state và tự động chọn cặp pokemon để ghép đôi.

**Pipeline tổng quát:**
```
Game (Pygame)
    ↓  board state (numpy array)
Environment Wrapper (Gymnasium)
    ↓  obs, reward, done
AI Agent (agent.py)
    ↓  state → action
Neural Network Model (model.py)
    ↓  training signal
Training Loop (train.py)
    ↓  saves checkpoint
logs/checkpoints/best.pt
```

---

## 👥 Phân công nhóm

| # | Vai trò | Nhiệm vụ chính | Module phụ trách |
|---|---|---|---|
| 1 | 🧑‍💻 **AI Architect** | Thiết kế kiến trúc hệ thống, quản lý Git, review & resolve conflict | Toàn bộ repo |
| 2 | 🎮 **Game Environment Developer** | Chuyển đổi Pygame sang AI Environment | `environment/` |
| 3 | 🔢 **Data & State Representation Engineer** | Xử lý State và Action, trích xuất dữ liệu game | `environment/`, `utils/` |
| 4 | 🧠 **Deep Learning Engineer** | Xây dựng CNN Model PyTorch | `model/` |
| 5 | 🤖 **RL Agent Developer** | Thuật toán Q-Learning, Replay Memory, epsilon-greedy | `agent/` |
| 6 | 📊 **Training, Tuning & Visualization** | Huấn luyện, tinh chỉnh siêu tham số, trực quan hóa | `training/` |

### Chi tiết nhiệm vụ từng thành viên

#### 🧑‍💻 Người 1 — AI Architect (Trưởng nhóm)
- Phân tích source code gốc `Pikachu-Matching-Game` để hiểu cách game vẽ map và thuật toán tìm đường (BFS/DFS).
- Thiết kế kiến trúc thư mục (tách riêng `game`, `agent.py`, `model.py`, `train.py`).
- Thiết lập repository chung, review code, resolve conflict khi ghép code của các thành viên.

#### 🎮 Người 2 — Game Environment Developer
- Xóa bỏ hệ thống lắng nghe sự kiện chuột/bàn phím (`pygame.event.get()`).
- Viết lại hàm khởi tạo `reset()` để tạo map mới ngẫu nhiên mỗi vòng chơi.
- Viết hàm `play_step(action)` nhận tọa độ từ AI, thực thi click, gọi thuật toán check đường dẫn nội tại của game, và trả về bộ 4 biến: `(state, reward, done, score)`.

#### 🔢 Người 3 — Data & State Representation Engineer
- Viết các hàm helper để trích xuất trạng thái ván game thành **ma trận số** (numpy array / PyTorch tensor) để nạp vào Model.
- Viết thuật toán `get_valid_actions(state)` — tìm tất cả các cặp có thể ăn được trên bàn — để giới hạn/hỗ trợ AI không chọn nước đi bị chặn (**action masking**).
- Đảm nhiệm việc chuyển đổi dữ liệu qua lại giữa môi trường Game và Agent.

#### 🧠 Người 4 — Deep Learning Engineer
- Thiết kế **CNN (Convolutional Neural Network)** trong `model.py` để xử lý đầu vào là ma trận map.
- Cấu trúc mạng: `Input Layer` → `Conv2D Layers` → `Flatten` → `Dense (Linear) Layers` → `Output (Action Q-Values)`.
- Viết hàm `forward`, hàm tính **Loss** (Mean Squared Error) và chọn **Optimizer** (Adam).

#### 🤖 Người 5 — RL Agent Developer
- Tạo `agent.py`. Quản lý `deque` để lưu **Replay Memory** `(state, action, reward, next_state, done)`.
- Implement `get_action(state)` kết hợp **epsilon-greedy** (sinh ngẫu nhiên ban đầu, giảm dần epsilon để AI dùng model suy luận về sau).
- Viết `train_short_memory` (học ngay sau mỗi lượt click) và `train_long_memory` (lấy ngẫu nhiên batch từ Memory sau mỗi ván game).

#### 📊 Người 6 — Training, Tuning & Visualization
- Viết **vòng lặp huấn luyện chính** trong `train.py` (`while True:`).
- Tinh chỉnh **hyperparameters**: Learning rate, tốc độ giảm epsilon, batch size, discount factor (gamma).
- Dùng **matplotlib** hoặc **TensorBoard** vẽ biểu đồ tiến bộ AI (trục X: số ván, trục Y: điểm số / số cặp ăn được).
- Tích hợp model đã train vào UI để quay **video demo**. Điều chỉnh `pygame clock` (FPS) để tăng tốc khi train, làm chậm khi demo.

---

## 🗂️ Cấu trúc thư mục

```
pokemon-auto-bot/GAME/
│
├── 📁 Pikachu-Matching-Game/   ← Game gốc (KHÔNG sửa trực tiếp)
│   ├── pikachu.py              ← Main game engine
│   ├── fix_grid.py             ← Tiện ích sửa lưới
│   ├── requirements.txt        ← Dependencies game
│   ├── Resources/              ← Sprites, âm thanh, hình nền
│   └── User_data/              ← Dữ liệu người chơi, bảng xếp hạng
│
├── 📁 environment/             ← 🔌 Cầu nối game ↔ AI Agent
│   ├── __init__.py
│   └── env.py                  ← PokemonEnv (Gymnasium interface)
│
├── 📁 agent/                   ← 🤖 Logic ra quyết định của AI
│   ├── __init__.py
│   └── agent.py                ← PokemonAgent (select_action, run_episode)
│
├── 📁 model/                   ← 🧠 Kiến trúc Neural Network
│   ├── __init__.py
│   └── model.py                ← PokemonModel (CNN/DNN/Transformer)
│
├── 📁 training/                ← 🏋️ Vòng lặp huấn luyện
│   ├── __init__.py
│   └── train.py                ← train(), evaluate(), logging
│
├── 📁 configs/                 ← ⚙️ Cấu hình hyperparameters
│   └── agent_config.yaml       ← Tất cả config: model, agent, training, env
│
├── 📁 logs/                    ← 📊 Output của quá trình training
│   ├── checkpoints/            ← Model weights (.pt files)
│   ├── tensorboard/            ← TensorBoard event files
│   └── runs/                   ← Lịch sử training runs
│
├── 📁 data/                    ← 🗃️ Dữ liệu (nếu dùng imitation learning)
│   ├── raw/                    ← Dữ liệu thô (game replays)
│   └── processed/              ← Dữ liệu đã xử lý (tensors)
│
├── 📁 tests/                   ← 🧪 Unit tests
│   ├── __init__.py
│   ├── test_agent/             ← Test cho agent module
│   ├── test_model/             ← Test cho model module
│   └── test_env/               ← Test cho environment module
│
├── 📁 docs/                    ← 📚 Tài liệu dự án
│   └── diagrams/               ← Sơ đồ kiến trúc, flowchart
│
├── 📁 scripts/                 ← 🛠️ Các script tiện ích
│   └── .gitkeep
│
├── 📁 game/                    ← 🎮 (Dành cho Game Dev custom thêm tính năng)
│   └── .gitkeep
│
├── main.py                     ← 🚀 Entry point chính
├── requirements.txt            ← Dependencies toàn bộ project
├── .gitignore                  ← File loại trừ khỏi Git
└── README.md                   ← 📖 File này
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Clone repo & cài đặt môi trường

```bash
# Clone repo
git clone <repository-url>
cd pokemon-auto-bot/GAME

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy game gốc (kiểm tra game hoạt động)

```bash
cd Pikachu-Matching-Game
python pikachu.py
```

### 3. Huấn luyện AI Agent

```bash
# Từ thư mục gốc GAME/
python main.py --mode train --config configs/agent_config.yaml
```

### 4. Demo AI tự chơi

```bash
python main.py --mode play --checkpoint logs/checkpoints/best.pt
```

### 5. Đánh giá Agent

```bash
python main.py --mode eval --checkpoint logs/checkpoints/best.pt
```

### 6. Xem TensorBoard

```bash
tensorboard --logdir logs/tensorboard
# Mở trình duyệt: http://localhost:6006
```

---

## 🧠 Thuật toán AI

Dự án hỗ trợ (hoặc sẽ implement) các thuật toán:

| Thuật toán | Mô tả | Config key |
|---|---|---|
| **DQN** | Deep Q-Network — off-policy, dùng replay buffer | `algorithm: "DQN"` |
| **DDQN** | Double DQN — giảm overestimate Q-value | `algorithm: "DDQN"` |
| **PPO** | Proximal Policy Optimization — stable, on-policy | `algorithm: "PPO"` |
| **A2C** | Advantage Actor-Critic — đơn giản, nhanh | `algorithm: "A2C"` |

Chỉnh thuật toán trong file `configs/agent_config.yaml`:
```yaml
agent:
  algorithm: "DQN"   # Đổi thành PPO, A2C, DDQN tùy ý
```

---

## 📐 Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (Entry Point)                 │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    [play mode]    [train mode]    [eval mode]
          │              │
          └──────┬───────┘
                 ▼
    ┌─────────────────────┐
    │   agent/agent.py    │  ← Ra quyết định
    │   PokemonAgent      │
    └──────┬──────────────┘
           │ state → action
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────┐
│ model/  │  │  environment/    │
│model.py │  │  env.py          │
│Neural   │  │  PokemonEnv      │
│Network  │  │  (Gym wrapper)   │
└─────────┘  └────────┬─────────┘
                      │ wraps
               ┌──────▼──────┐
               │  Pikachu-   │
               │  Matching-  │
               │  Game/      │
               │ pikachu.py  │
               └─────────────┘
```

---

## 📏 Quy tắc làm việc nhóm (Git Workflow)

### Branch naming
```
main          ← Production, chỉ merge khi đã test xong
develop       ← Development branch chính
feature/env   ← Branch cho Environment module
feature/agent ← Branch cho Agent module
feature/model ← Branch cho Model module
feature/train ← Branch cho Training module
```

### Quy trình làm việc
```bash
# 1. Tạo branch mới từ develop
git checkout develop
git pull origin develop
git checkout -b feature/your-feature

# 2. Code + commit thường xuyên
git add .
git commit -m "feat(agent): implement epsilon-greedy policy"

# 3. Push và tạo Pull Request
git push origin feature/your-feature
# → Tạo PR vào develop trên GitHub

# 4. Code review (ít nhất 1 người review trước khi merge)
```

### Commit message convention
```
feat(module):  thêm tính năng mới
fix(module):   sửa bug
docs(module):  cập nhật tài liệu
test(module):  thêm/sửa tests
refactor(module): refactor code
chore:         cập nhật config, dependencies
```

Ví dụ:
```
feat(agent): implement DQN agent with replay buffer
fix(env): fix reward calculation when board is cleared
docs(readme): add training instructions
test(model): add unit tests for CNN forward pass
```

---

## 🧪 Chạy Tests

```bash
# Chạy toàn bộ tests
pytest tests/

# Chạy test với coverage report
pytest tests/ --cov=. --cov-report=html

# Chạy test cho 1 module cụ thể
pytest tests/test_agent/
pytest tests/test_model/
pytest tests/test_env/
```

---

## ⚙️ Cấu hình (configs/agent_config.yaml)

Tất cả hyperparameters được quản lý tập trung tại [`configs/agent_config.yaml`](configs/agent_config.yaml).

Các section chính:
- `agent`: epsilon, gamma, batch_size, replay buffer
- `model`: kiến trúc CNN, số layers, dropout
- `environment`: kích thước board, headless mode, time limit
- `training`: số episodes, learning rate, tần suất checkpoint
- `paths`: đường dẫn output

---

## 📦 Dependencies chính

| Thư viện | Mục đích |
|---|---|
| `pygame` | Game engine |
| `torch` | Deep Learning (PyTorch) |
| `gymnasium` | RL Environment interface |
| `stable-baselines3` | Các thuật toán RL có sẵn |
| `tensorboard` | Visualize training |
| `numpy` | Xử lý ma trận board state |
| `pyyaml` | Đọc config file |
| `pytest` | Unit testing |

---

## 📚 Tài liệu tham khảo

- [Deep Q-Network (DQN) Paper](https://arxiv.org/abs/1312.5602) — DeepMind
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — OpenAI
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 👨‍💻 Nhóm phát triển

> Dự án Trí Tuệ Nhân Tạo 

| Thành viên | MSSV | Vai trò |
|---|---|---|
| Lê Nguyên Khôi | 030239230005 |  AI Architect |
| Nguyễn Phan Quỳnh Thy | 030239230241 | Game Environment Developer  |
| Đào Việt Anh| 030239230093 | Trưởng nhóm / Data & State Representation Engineer |
| Trần Thị Trúc Xinh | 030239230298 | Deep Learning Engineer |
| Nguyễn Khánh Dương | 030239230036 | RL Agent Developer|
| Nguyễn Thị Nhã Phương | 030239230194 | Training, Tuning & Visualization |

---

*📅 Last updated: June 2026*
