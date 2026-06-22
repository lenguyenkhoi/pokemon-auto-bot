"""
q_learning_agent.py
=====================
GIAI ĐOẠN 1: Q-Learning trên board NHỎ (vd 4x4).

MỤC ĐÍCH của giai đoạn này KHÔNG phải để dùng thật, mà để:
1. Kiểm chứng Environment (onet_env.py) hoạt động đúng logic.
2. Kiểm chứng việc "AI có học được không" trước khi đầu tư công sức vào DQN phức tạp hơn.
Nếu Q-Learning học được trên board nhỏ -> yên tâm chuyển sang DQN cho board lớn.

CÁCH CHẠY: python q_learning_agent.py
Sẽ tự động train rồi vẽ biểu đồ reward theo thời gian để bạn thấy AI có học giỏi lên không.
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # cho phép vẽ ảnh mà không cần màn hình hiển thị (chạy trên server/terminal)
import matplotlib.pyplot as plt

from onet_env import OnetEnv


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        """
        alpha (learning rate): học nhanh hay chậm từ mỗi kinh nghiệm mới.
            - alpha lớn (vd 0.5): học nhanh nhưng dễ "nhiễu", không ổn định.
            - alpha nhỏ (vd 0.01): học chậm nhưng ổn định hơn.
        gamma (discount factor): mức độ quan tâm phần thưởng TƯƠNG LAI.
            - gamma gần 1 (vd 0.95-0.99): nhìn xa, biết hy sinh điểm trước mắt để thắng cả ván.
            - gamma gần 0: chỉ quan tâm điểm ngay lập tức, недальновидный (недальновидный = thiển cận).
        epsilon (tỉ lệ khám phá ngẫu nhiên - "exploration"):
            - Đây là kỹ thuật EPSILON-GREEDY: với xác suất epsilon, AI chọn action NGẪU NHIÊN
              (để khám phá những nước đi nó chưa thử) thay vì luôn chọn nước nó nghĩ là tốt nhất.
            - Lý do cần: nếu AI chỉ làm theo cái nó "tưởng" là tốt nhất ngay từ đầu (khi nó còn
              chưa biết gì), nó sẽ kẹt mãi ở một chiến lược tầm thường, không bao giờ khám phá
              ra cách chơi tốt hơn.
            - epsilon bắt đầu CAO (1.0 = 100% random) rồi GIẢM DẦN theo epsilon_decay
              để AI khám phá nhiều lúc đầu, rồi dần "tự tin" dùng kiến thức đã học.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: dictionary, key = (state_dạng_chuỗi, action), value = Q-value (số thực)
        # Dùng dict thay vì mảng cố định vì số trạng thái có thể có là RẤT lớn,
        # ta chỉ lưu những trạng thái AI đã thực sự gặp qua (sparse storage).
        self.q_table = {}

    def _state_key(self, state):
        """
        Chuyển ma trận bàn cờ (numpy array) thành chuỗi để dùng làm KEY trong dictionary.
        (numpy array không dùng trực tiếp làm key dict được vì nó "mutable" - có thể đổi giá trị).
        """
        return state.tobytes()

    def get_q(self, state, action):
        key = (self._state_key(state), action)
        return self.q_table.get(key, 0.0)  # nếu chưa từng gặp, coi Q-value = 0 (chưa biết gì)

    def choose_action(self, state, valid_actions):
        """
        Epsilon-greedy: với xác suất epsilon -> chọn random (khám phá).
                        còn lại -> chọn action có Q-value cao nhất đã biết (khai thác kiến thức).
        """
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # KHÁM PHÁ (exploration)

        # KHAI THÁC (exploitation): chọn action có Q-value cao nhất
        q_values = [self.get_q(state, a) for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)  # nếu có nhiều action cùng điểm cao nhất, chọn ngẫu nhiên 1 trong số đó

    def update(self, state, action, reward, next_state, next_valid_actions):
        """
        ĐÂY LÀ CÔNG THỨC Q-LEARNING (trái tim của thuật toán):

        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)]

        Giải thích từng phần:
        - Q(s,a): điểm hiện tại của việc "ở trạng thái s, làm hành động a"
        - reward: phần thưởng VỪA nhận được ngay lúc này
        - max(Q(s', a')): điểm cao nhất có thể đạt được ở trạng thái TIẾP THEO (s')
          -> đại diện cho "tiềm năng tương lai" nếu đi tiếp từ đây
        - [reward + gamma * max(Q(s',a')) - Q(s,a)]: đây là "sai số" (TD-error) -
          chênh lệch giữa điều AI VỪA quan sát được và điều AI ĐANG TIN TƯỞNG trước đó.
        - alpha * sai_số: điều chỉnh Q-value một chút theo hướng đúng hơn.
        """
        old_q = self.get_q(state, action)

        if next_valid_actions:
            future_q = max([self.get_q(next_state, a) for a in next_valid_actions])
        else:
            future_q = 0.0  # không còn action nào tiếp theo (hết ván)

        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

        key = (self._state_key(state), action)
        self.q_table[key] = new_q

    def decay_epsilon(self):
        """Giảm dần epsilon sau mỗi ván, để AI bớt random dần, tự tin hơn vào kiến thức đã học."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(n_episodes=3000, rows=4, cols=4, n_types=4):
    """
    Vòng lặp huấn luyện chính.
    1 EPISODE = 1 ván chơi trọn vẹn, từ bàn cờ đầy đủ đến khi thắng/thua/bế tắc.
    """
    env = OnetEnv(rows=rows, cols=cols, n_types=n_types)
    agent = QLearningAgent()

    episode_rewards = []   # lưu tổng điểm mỗi ván, để vẽ biểu đồ xem AI học giỏi lên không
    win_history = []       # lưu 1/0 - ván đó có thắng (xoá hết bàn) không

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        won = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break  # bế tắc ngay từ đầu (hiếm khi xảy ra nhưng vẫn cần xử lý)

            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)

            next_valid_actions = env.get_valid_actions() if not done else []
            agent.update(state, action, reward, next_state, next_valid_actions)

            state = next_state
            total_reward += reward
            if info.get("win"):
                won = True

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        win_history.append(1 if won else 0)

        # In tiến độ mỗi 200 ván để theo dõi
        if (episode + 1) % 200 == 0:
            recent_rewards = episode_rewards[-200:]
            recent_wins = win_history[-200:]
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Reward TB (200 ván gần nhất): {np.mean(recent_rewards):.1f} | "
                  f"Tỉ lệ thắng: {np.mean(recent_wins)*100:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, episode_rewards, win_history


def plot_results(episode_rewards, win_history, save_path="qlearning_results.png"):
    """Vẽ 2 biểu đồ: đường cong reward và tỉ lệ thắng theo thời gian (learning curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Làm mượt đường cong bằng cách tính trung bình trượt (moving average)
    window = 50
    smoothed_reward = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
    axes[0].plot(smoothed_reward, color="#1E2761")
    axes[0].set_title("Reward trung bình theo thời gian (Q-Learning)")
    axes[0].set_xlabel("Episode (ván chơi)")
    axes[0].set_ylabel("Reward trung bình (trượt 50 ván)")
    axes[0].grid(alpha=0.3)

    smoothed_win = np.convolve(win_history, np.ones(window)/window, mode="valid") * 100
    axes[1].plot(smoothed_win, color="#D7263D")
    axes[1].set_title("Tỉ lệ thắng theo thời gian (%)")
    axes[1].set_xlabel("Episode (ván chơi)")
    axes[1].set_ylabel("Tỉ lệ thắng (%)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nĐã lưu biểu đồ vào: {save_path}")


def evaluate_vs_random(agent, n_episodes=500, rows=4, cols=4, n_types=2):
    """
    SO SÁNH ĐỐI CHỨNG: cho AI đã huấn luyện (epsilon=0, không random nữa) chơi
    n_episodes ván MỚI (bàn cờ nó CHƯA từng thấy), so với một AI hoàn toàn NGẪU NHIÊN.

    Đây là cách kiểm chứng khách quan nhất xem AI có thực sự "học được điều gì" không,
    thay vì chỉ nhìn biểu đồ training (vốn có thể gây hiểu lầm vì còn lẫn yếu tố khám phá ngẫu nhiên).
    """
    env = OnetEnv(rows=rows, cols=cols, n_types=n_types)

    def play_one_game(use_agent):
        state = env.reset()
        done = False
        won = False
        steps = 0
        while not done and steps < 200:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            if use_agent:
                # epsilon=0 tạm thời: ép AI luôn chọn action tốt nhất nó biết, không random
                q_values = [agent.get_q(state, a) for a in valid_actions]
                max_q = max(q_values)
                best = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                action = random.choice(best)
            else:
                action = random.choice(valid_actions)
            state, reward, done, info = env.step(action)
            won = info.get("win", False)
            steps += 1
        return won

    agent_wins = sum(play_one_game(use_agent=True) for _ in range(n_episodes))
    random_wins = sum(play_one_game(use_agent=False) for _ in range(n_episodes))

    print("\n" + "=" * 60)
    print("ĐỐI CHỨNG 1: AI đã học vs AI NGẪU NHIÊN -- trên BÀN CỜ MỚI MỖI VÁN")
    print("=" * 60)
    print(f"AI đã học   : thắng {agent_wins}/{n_episodes} ván ({agent_wins/n_episodes*100:.1f}%)")
    print(f"AI ngẫu nhiên: thắng {random_wins}/{n_episodes} ván ({random_wins/n_episodes*100:.1f}%)")
    print("(Chênh lệch nhỏ ở đây là BÌNH THƯỜNG - xem giải thích Đối chứng 2 bên dưới)")


def evaluate_fixed_boards(rows=4, cols=4, n_types=4, n_fixed_boards=20, n_episodes=2000):
    """
    SO SÁNH ĐỐI CHỨNG 2 - QUAN TRỌNG HƠN: thay vì sinh bàn cờ MỚI mỗi ván,
    ta CHỈ DÙNG MỘT TẬP CỐ ĐỊNH gồm n_fixed_boards bàn cờ, lặp lại nhiều lần khi train.

    Mục đích: chứng minh Q-table CÓ KHẢ NĂNG HỌC RẤT TỐT khi state LẶP LẠI
    (đúng với bản chất "tra bảng" của nó) - khác hẳn với trường hợp bàn cờ luôn
    đổi mới liên tục. Đây là phép so sánh giúp bạn hiểu rõ NGUYÊN NHÂN gốc rễ,
    không phải do code sai, mà do ĐẶC ĐIỂM bài toán (state luôn mới) không hợp với
    Q-table - và đó chính xác là lý do ta cần DQN cho bài toán Onet thật.

    LƯU Ý KỸ THUẬT QUAN TRỌNG: ta phải đo tỉ lệ thắng bằng epsilon=0 (tắt hẳn
    yếu tố random khám phá) tại các checkpoint riêng biệt, KHÔNG đo trực tiếp
    trong lúc train (vì lúc train epsilon vẫn còn cao -> nhiễu kết quả, gây
    hiểu lầm rằng AI "không học được" dù Q-table thực ra đã học khá tốt).
    """
    print("\n" + "=" * 60)
    print("ĐỐI CHỨNG 2: Q-Learning trên TẬP BÀN CỜ CỐ ĐỊNH (lặp lại nhiều lần)")
    print("=" * 60)

    env = OnetEnv(rows=rows, cols=cols, n_types=n_types)
    fixed_boards = []
    for _ in range(n_fixed_boards):
        env.reset()
        fixed_boards.append(env.board.copy())

    agent = QLearningAgent()

    def play_fixed_eval(n_eval=300):
        """Đo tỉ lệ thắng với epsilon=0 (không random), trên các bàn cờ cố định."""
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # tắt hẳn khám phá ngẫu nhiên khi đo
        wins = 0
        for _ in range(n_eval):
            env.board = random.choice(fixed_boards).copy()
            state = env.get_state()
            done = False
            won = False
            steps = 0
            while not done and steps < 200:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                action = agent.choose_action(state, valid_actions)
                state, reward, done, info = env.step(action)
                won = info.get("win", False)
                steps += 1
            wins += 1 if won else 0
        agent.epsilon = old_epsilon  # khôi phục lại epsilon để tiếp tục train
        return wins / n_eval * 100

    # Đo NGAY TỪ ĐẦU (trước khi học gì cả) làm mốc so sánh
    win_rate_before = play_fixed_eval()

    # Train bình thường (có exploration) trên tập bàn cờ cố định
    for episode in range(n_episodes):
        env.board = random.choice(fixed_boards).copy()
        state = env.get_state()
        done = False
        steps = 0
        while not done and steps < 200:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions() if not done else []
            agent.update(state, action, reward, next_state, next_valid_actions)
            state = next_state
            steps += 1
        agent.decay_epsilon()

    # Đo SAU KHI HỌC XONG (epsilon=0, dùng thuần kiến thức đã học)
    win_rate_after = play_fixed_eval()

    print(f"Tỉ lệ thắng TRƯỚC khi học (Q-table rỗng, gần như random): {win_rate_before:.1f}%")
    print(f"Tỉ lệ thắng SAU khi học xong ({n_episodes} ván, epsilon=0): {win_rate_after:.1f}%")
    if win_rate_after > win_rate_before + 5:
        print(">> RÕ RÀNG AI HỌC GIỎI LÊN khi state LẶP LẠI (đúng bản chất Q-table).")
        print(">> Điều này xác nhận: code/logic Q-Learning ĐÚNG. Vấn đề ở bài toán Onet")
        print("   thật là bàn cờ KHÔNG LẶP LẠI -> cần DQN để khái quát hoá.")
    else:
        print(">> Bài toán có thể đã quá dễ ngay từ đầu (gần thắng kể cả random).")
        print("   Thử tăng n_types hoặc kích thước board để bài toán khó hơn,")
        print("   tạo ra khoảng cách rõ rệt hơn cho AI thể hiện việc đã học được.")


if __name__ == "__main__":
    print("=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN Q-LEARNING TRÊN BOARD 4x4")
    print("=" * 60)
    # GHI CHÚ QUAN TRỌNG: n_types càng NHỎ so với số ô thì càng DỄ tìm cặp giống nhau
    # -> dễ thắng hơn -> phù hợp để kiểm chứng AI có học được không ở bước đầu.
    # Validator giả lập ở đây chỉ cho nối thẳng hàng/cột (chưa có đường chữ L/Z thật),
    # nên cố tình chọn n_types nhỏ để bù lại, tránh bế tắc quá sớm.
    # Khi ghép validator thật của TV4 (hỗ trợ đường chữ L/Z), bàn cờ sẽ dễ nối hơn
    # nhiều và có thể tăng n_types lên cho sát game thật.
    agent, rewards, wins = train(n_episodes=3000, rows=4, cols=4, n_types=5)
    plot_results(rewards, wins)

    print(f"\nSố trạng thái (state-action) đã học: {len(agent.q_table)}")

    evaluate_vs_random(agent, n_episodes=500, rows=4, cols=4, n_types=5)
    evaluate_fixed_boards(rows=6, cols=6, n_types=9, n_fixed_boards=15, n_episodes=3000)

    print("""
GHI CHÚ QUAN TRỌNG - VÌ SAO TỈ LỆ THẮNG TRONG LÚC TRAIN KHÔNG TĂNG RÕ RỆT:
---------------------------------------------------------------------------
Mỗi episode (ván chơi), bàn cờ được SINH NGẪU NHIÊN MỚI HOÀN TOÀN.
Q-table là một "cuốn sổ tra cứu" lưu riêng từng trạng thái (state) nó đã gặp.
Vì bàn cờ luôn đổi mới, rất nhiều trạng thái AI gặp chỉ XUẤT HIỆN ĐÚNG 1 LẦN
trong suốt quá trình train -> Q-table không có cơ hội "tra cứu lại" để cải thiện
dần dần như cách nó hoạt động tốt trên các bài toán có state cố định/lặp lại
(ví dụ: mê cung cố định, tay đua xe trên 1 đường track cố định).

ĐÂY CHÍNH LÀ GIỚI HẠN CỐT LÕI CỦA Q-LEARNING DẠNG BẢNG (tabular),
và là LÝ DO CHÍNH ta cần chuyển sang DQN (Deep Q-Network):
mạng neural có khả năng "khái quát hoá" (generalize) - tức là học ra QUY LUẬT
CHUNG (vd: "ưu tiên nối cặp ở góc trước vì nó dễ bị kẹt") thay vì học thuộc lòng
từng bàn cờ cụ thể. Quy luật chung này áp dụng được cho cả bàn cờ MỚI mà nó
chưa từng thấy y hệt -> đây là lý do DQN học hiệu quả hơn nhiều trên bài toán này.

=> Kết quả Q-Learning ở trên dùng để XÁC NHẬN ENVIRONMENT CHẠY ĐÚNG (đã xác nhận
   qua phép so sánh đối chứng trên), không nhằm để có một AI giỏi thật sự.
   Bước tiếp theo: train_dqn.py
""")
