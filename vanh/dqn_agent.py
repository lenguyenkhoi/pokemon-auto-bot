"""
dqn_agent.py
=============
GIAI ĐOẠN 2: Deep Q-Network (DQN) trên board kích thước THẬT (vd 8x6 = 48 ô).

KHÁC BIỆT SO VỚI Q-LEARNING:
- Thay vì tra bảng (Q-table), ta dùng MẠNG NEURAL để DỰ ĐOÁN Q-value.
- Mạng neural nhận vào: (trạng thái bàn cờ, một action cụ thể) -> trả ra: điểm Q-value
  ước lượng cho việc thực hiện action đó tại trạng thái đó.
- Vì mạng học ra QUY LUẬT chung (không học thuộc lòng từng bàn cờ), nó tổng quát hoá
  được sang các bàn cờ mới chưa từng thấy -> đây là điều Q-table không làm được.

HAI KỸ THUẬT QUAN TRỌNG GIÚP DQN HỌC ỔN ĐỊNH:
1. EXPERIENCE REPLAY: lưu lại các "kinh nghiệm" (state, action, reward, next_state, done)
   vào một bộ nhớ đệm (replay buffer). Khi học, LẤY NGẪU NHIÊN một batch ra để học,
   thay vì học theo đúng thứ tự xảy ra. Lý do: các bước liên tiếp trong 1 ván rất
   giống nhau (tương quan cao) -> nếu học liên tục theo thứ tự, mạng dễ bị "lệch"
   theo xu hướng tạm thời thay vì học quy luật tổng quát.
2. TARGET NETWORK: dùng 2 bản sao của mạng - một mạng "main" cập nhật liên tục mỗi
   bước, một mạng "target" chỉ copy từ main sau mỗi N bước. Mạng target dùng để TÍNH
   điểm kỳ vọng tương lai khi cập nhật main. Nếu dùng chung 1 mạng cho cả 2 việc,
   "mục tiêu" mà ta đang cố học sẽ liên tục thay đổi theo từng bước -> khó hội tụ
   (giống việc vừa chạy vừa đuổi theo một đích đang di chuyển ngẫu nhiên).

CÁCH CHẠY: python dqn_agent.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from onet_env import OnetEnv


# ======================================================================
# BƯỚC 1: BIỂU DIỄN STATE + ACTION THÀNH VECTOR SỐ ĐỂ ĐƯA VÀO MẠNG NEURAL
# ======================================================================
def encode_state_action(board, action, n_types):
    """
    Mạng neural chỉ hiểu số, không hiểu "bàn cờ" hay "toạ độ" trực tiếp.
    Ta cần biến (board, action) thành một VECTOR SỐ cố định độ dài.

    Cách làm:
    - One-hot encode bàn cờ: mỗi ô được biểu diễn bằng (n_types + 1) số 0/1
      (cộng thêm 1 để biểu diễn trạng thái "ô trống").
      Ví dụ: nếu có 5 loại Pokémon, ô chứa loại số 2 sẽ là [0,0,1,0,0,0],
      ô trống sẽ là [0,0,0,0,0,1].
    - Thêm vào cuối: vị trí của 2 ô trong action, dạng one-hot theo hàng + cột.

    Đây không phải cách DUY NHẤT để encode (có thể dùng CNN xử lý ảnh bàn cờ
    trực tiếp dạng ma trận 2D), nhưng cách one-hot + MLP (mạng kết nối đầy đủ)
    này ĐƠN GIẢN HƠN NHIỀU để cài đặt đúng trong thời gian ngắn, và đủ tốt
    cho bài toán board cỡ vài chục ô.
    """
    rows, cols = board.shape
    n_classes = n_types + 1  # +1 cho "ô trống"

    # One-hot toàn bộ bàn cờ -> vector dài rows*cols*n_classes
    board_flat = board.flatten()
    board_onehot = np.zeros((rows * cols, n_classes), dtype=np.float32)
    for i, val in enumerate(board_flat):
        idx = n_types if val == -1 else val  # -1 (trống) map vào index cuối cùng
        board_onehot[i, idx] = 1.0
    board_vec = board_onehot.flatten()

    # One-hot vị trí 2 ô trong action
    (r1, c1), (r2, c2) = action
    pos_vec = np.zeros(rows + cols + rows + cols, dtype=np.float32)
    pos_vec[r1] = 1.0
    pos_vec[rows + c1] = 1.0
    pos_vec[rows + cols + r2] = 1.0
    pos_vec[rows + cols + rows + c2] = 1.0

    return np.concatenate([board_vec, pos_vec])


def get_input_dim(rows, cols, n_types):
    n_classes = n_types + 1
    return rows * cols * n_classes + 2 * (rows + cols)


# ======================================================================
# BƯỚC 2: KIẾN TRÚC MẠNG NEURAL
# ======================================================================
class QNetwork(nn.Module):
    """
    Mạng đơn giản: vài lớp Linear (fully-connected) + hàm kích hoạt ReLU.
    Input: vector (state, action) đã encode.
    Output: 1 số thực = Q-value ước lượng.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ======================================================================
# BƯỚC 3: REPLAY BUFFER (bộ nhớ kinh nghiệm)
# ======================================================================
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)  # deque tự động bỏ kinh nghiệm cũ nhất khi đầy

    def push(self, state, action, reward, next_state, next_valid_actions, done):
        self.buffer.append((state, action, reward, next_state, next_valid_actions, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# BƯỚC 4: DQN AGENT - GHÉP TẤT CẢ LẠI
# ======================================================================
class DQNAgent:
    def __init__(self, rows, cols, n_types, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 target_update_freq=200):
        self.rows = rows
        self.cols = cols
        self.n_types = n_types
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = get_input_dim(rows, cols, n_types)

        # main_net: mạng học liên tục mỗi bước
        self.main_net = QNetwork(input_dim).to(self.device)
        # target_net: bản sao "đông cứng", chỉ copy từ main_net định kỳ -> giúp ổn định huấn luyện
        self.target_net = QNetwork(input_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()  # target_net không cần tính gradient, chỉ dùng để dự đoán

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

    def _q_values_for_actions(self, net, board, actions):
        """Tính Q-value cho TỪNG action trong danh sách actions, dùng mạng `net`."""
        if not actions:
            return torch.tensor([], device=self.device)
        vectors = [encode_state_action(board, a, self.n_types) for a in actions]
        batch = torch.tensor(np.array(vectors), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = net(batch).squeeze(-1)
        return q_values

    def choose_action(self, board, valid_actions):
        """Epsilon-greedy: random để khám phá, hoặc chọn Q-value cao nhất để khai thác."""
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self._q_values_for_actions(self.main_net, board, valid_actions)
        best_idx = torch.argmax(q_values).item()
        return valid_actions[best_idx]

    def remember(self, state, action, reward, next_state, next_valid_actions, done):
        self.replay_buffer.push(state, action, reward, next_state, next_valid_actions, done)

    def train_step(self, batch_size=64):
        """
        Lấy 1 batch ngẫu nhiên từ replay buffer, cập nhật mạng main_net.
        Đây là bước tương đương công thức Q-Learning, nhưng thay vì cập nhật 1 ô
        trong bảng, ta cập nhật TRỌNG SỐ MẠNG NEURAL bằng gradient descent.
        """
        if len(self.replay_buffer) < batch_size:
            return None  # chưa đủ dữ liệu để học

        batch = self.replay_buffer.sample(batch_size)

        # Với mỗi kinh nghiệm trong batch, tính "target" (giá trị Q mong muốn đạt được)
        state_action_vectors = []
        targets = []

        for state, action, reward, next_state, next_valid_actions, done in batch:
            vec = encode_state_action(state, action, self.n_types)
            state_action_vectors.append(vec)

            if done or not next_valid_actions:
                target = reward  # không còn tương lai để cộng thêm
            else:
                # Dùng TARGET NETWORK (không phải main_net) để tính điểm tương lai tốt nhất
                future_q = self._q_values_for_actions(self.target_net, next_state, next_valid_actions)
                target = reward + self.gamma * torch.max(future_q).item()
            targets.append(target)

        inputs = torch.tensor(np.array(state_action_vectors), dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device).unsqueeze(1)

        predictions = self.main_net(inputs)
        loss = self.loss_fn(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        # Định kỳ copy trọng số từ main_net sang target_net
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Lưu model đã train để dùng SUY LUẬN (không train) khi chạy trên app thật."""
        torch.save({
            "model_state_dict": self.main_net.state_dict(),
            "rows": self.rows,
            "cols": self.cols,
            "n_types": self.n_types,
        }, path)
        print(f"Đã lưu model vào: {path}")

    @classmethod
    def load(cls, path, device=None):
        """Load model đã train, dùng để CHƠI THẬT (chỉ predict, không train nữa)."""
        checkpoint = torch.load(path, map_location=device or "cpu")
        agent = cls(checkpoint["rows"], checkpoint["cols"], checkpoint["n_types"])
        agent.main_net.load_state_dict(checkpoint["model_state_dict"])
        agent.main_net.eval()
        agent.epsilon = 0.0  # khi chơi thật, KHÔNG random nữa, luôn chọn nước tốt nhất đã học
        return agent

    def predict(self, board, valid_actions):
        """
        HÀM DÙNG KHI CHẠY TRÊN APP THẬT (không train, chỉ suy luận).
        Input: board (ma trận từ module CV của TV1), valid_actions (từ module TV4).
        Output: action tốt nhất (cellA, cellB), hoặc None nếu không có action nào.
        """
        if not valid_actions:
            return None
        q_values = self._q_values_for_actions(self.main_net, board, valid_actions)
        best_idx = torch.argmax(q_values).item()
        return valid_actions[best_idx]


# ======================================================================
# BƯỚC 5: VÒNG LẶP HUẤN LUYỆN
# ======================================================================
def train_dqn(n_episodes=1500, rows=8, cols=6, n_types=8, batch_size=64):
    env = OnetEnv(rows=rows, cols=cols, n_types=n_types)
    agent = DQNAgent(rows=rows, cols=cols, n_types=n_types)

    episode_rewards = []
    win_history = []
    losses = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        won = False
        steps = 0

        while not done and steps < 300:  # giới hạn số bước để tránh vòng lặp vô hạn nếu có lỗi logic
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions() if not done else []

            agent.remember(state, action, reward, next_state, next_valid_actions, done)
            loss = agent.train_step(batch_size=batch_size)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1
            if info.get("win"):
                won = True

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        win_history.append(1 if won else 0)

        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_wins = win_history[-50:]
            avg_loss = np.mean(losses[-200:]) if losses else 0
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Reward TB: {np.mean(recent_rewards):.1f} | "
                  f"Tỉ lệ thắng: {np.mean(recent_wins)*100:.1f}% | "
                  f"Loss TB: {avg_loss:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, episode_rewards, win_history, losses


def plot_dqn_results(episode_rewards, win_history, losses, save_path="dqn_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    window = 30

    if len(episode_rewards) >= window:
        smoothed_reward = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(smoothed_reward, color="#1E2761")
    axes[0].set_title("Reward trung bình theo Episode (DQN)")
    axes[0].set_xlabel("Episode")
    axes[0].grid(alpha=0.3)

    if len(win_history) >= window:
        smoothed_win = np.convolve(win_history, np.ones(window)/window, mode="valid") * 100
        axes[1].plot(smoothed_win, color="#D7263D")
    axes[1].set_title("Tỉ lệ thắng theo Episode (%)")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.3)

    if losses:
        loss_window = min(200, len(losses))
        smoothed_loss = np.convolve(losses, np.ones(loss_window)/loss_window, mode="valid")
        axes[2].plot(smoothed_loss, color="#2C5F2D")
    axes[2].set_title("Loss theo bước huấn luyện")
    axes[2].set_xlabel("Training step")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nĐã lưu biểu đồ vào: {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN DQN TRÊN BOARD 8x6 (KÍCH THƯỚC GẦN THẬT)")
    print("=" * 60)
    print("Lưu ý: bước này tốn nhiều thời gian hơn Q-Learning vì phải train mạng neural.")
    print("Trên máy không có GPU, 1500 episode có thể mất 10-30 phút tuỳ cấu hình máy.\n")

    agent, rewards, wins, losses = train_dqn(n_episodes=1500, rows=8, cols=6, n_types=8)
    plot_dqn_results(rewards, wins, losses)

    agent.save("dqn_model.pt")

    print("""
BƯỚC TIẾP THEO:
- Xem file dqn_results.png: nếu đường tỉ lệ thắng có xu hướng ĐI LÊN theo thời gian
  (dù có dao động) và loss có xu hướng GIẢM hoặc ổn định, nghĩa là DQN đang học được.
- File dqn_model.pt chính là MODEL ĐÃ HUẤN LUYỆN, dùng để nộp bài và để TV5/TV6
  tích hợp vào pipeline chạy trên app thật, bằng cách:
      agent = DQNAgent.load("dqn_model.pt")
      action = agent.predict(board_tu_TV1, valid_actions_tu_TV4)
""")
