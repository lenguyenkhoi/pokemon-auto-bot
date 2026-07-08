import copy
import torch
import random
import numpy as np
from collections import deque

# Import các hằng số và hàm xử lý Action Masking từ Người số 3 gửi
from utils.state_utils import (
    get_valid_actions,
    extract_state,
    state_to_torch,
    batch_states_to_torch,
    qvalues_to_action,
)

# Cấu hình hằng số cho Agent
MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001

class PikachuAgent:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        """
        Khởi tạo Agent.
        - model: Mạng CNN do Người số 4 thiết kế trong model.py
        - optimizer: Bộ tối ưu hóa (ví dụ: Adam)
        - criterion: Hàm tính Loss (ví dụ: Mean Squared Error - MSE)
        - device: Thiết bị chạy PyTorch ('cpu' hoặc 'cuda')
        """
        self.n_games = 0
        self.epsilon = 1.0  # Tỷ lệ đi bừa (khám phá), sẽ giảm dần theo thời gian
        self.gamma = 0.9    # Hệ số chiết khấu (discount factor) cho phần thưởng tương lai
        self.memory = deque(maxlen=MAX_MEMORY)  # Bộ nhớ Replay Memory sử dụng deque
        self.training_steps = 0
        self.target_update_frequency = 50
        self.last_board_count = None
        
        # Thiết bị phần cứng
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optimizer
        self.criterion = criterion

    def remember(self, state, action, reward, next_state, done):
        """Lưu trữ trải nghiệm (S, A, R, S', done) vào bộ nhớ để học sau"""
        self.memory.append((state, action, reward, next_state, done))

    def _reward_from_transition(self, state, action, reward, next_state, done):
        """Tạo reward được làm giàu để agent học được mục tiêu clear board."""
        board_before = np.array(state, dtype=np.int32)
        board_after = np.array(next_state, dtype=np.int32)

        remaining_before = np.count_nonzero(board_before)
        remaining_after = np.count_nonzero(board_after)
        cleared = remaining_before - remaining_after

        shaped_reward = float(reward)
        if cleared > 0:
            shaped_reward += 10.0 * cleared
        if remaining_after == 0:
            shaped_reward += 100.0
        if done:
            shaped_reward += 25.0
        if reward < 0:
            shaped_reward -= 8.0
        return shaped_reward

    def train_short_memory(self, state, action, reward, next_state, done):
        """Học ngay lập tức sau mỗi lượt bấm (Single step training)"""
        shaped_reward = self._reward_from_transition(state, action, reward, next_state, done)

        s_dict = extract_state(state)
        next_s_dict = extract_state(next_state)
        s_tensors = state_to_torch(s_dict, device=self.device)
        next_s_tensors = state_to_torch(next_s_dict, device=self.device)
        self._train_step([s_tensors], [action], [shaped_reward], [next_s_tensors], [done])

    def train_long_memory(self):
        """Học từ quá khứ sau khi kết thúc một ván game bằng cách bốc ngẫu nhiên một Batch"""
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Bốc ngẫu nhiên một nhóm mẫu từ Replay Memory
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        
        # Phân tách dữ liệu từ mini_batch
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        s_dicts = [extract_state(s) for s in states]
        next_s_dicts = [extract_state(ns) for ns in next_states]
        s_tensors = batch_states_to_torch(s_dicts, device=self.device)
        next_s_tensors = batch_states_to_torch(next_s_dicts, device=self.device)

        shaped_rewards = [self._reward_from_transition(s, a, r, ns, d) for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones)]
        self._train_step(s_tensors, actions, shaped_rewards, next_s_tensors, dones, is_batch=True)

    def _train_step(self, states, actions, rewards, next_states, dones, is_batch=False):
        """Hàm logic huấn luyện DQN-style dùng target network để học từ reward."""
        if is_batch:
            state_input = states["board_onehot"].to(self.device)
            next_state_input = next_states["board_onehot"].to(self.device)
        else:
            state_input = states[0]["board_onehot"].unsqueeze(0).to(self.device)
            next_state_input = next_states[0]["board_onehot"].unsqueeze(0).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        self.model.train()
        pred = self.model(state_input)
        target = pred.detach().clone()

        with torch.no_grad():
            next_pred = self.target_model(next_state_input)

        H, W = state_input.shape[2], state_input.shape[3]
        batch_size = len(dones)
        for idx in range(batch_size):
            r1, c1, r2, c2 = actions[idx]
            idx1 = r1 * W + c1
            idx2 = r2 * W + c2

            q_target = rewards[idx].item()
            if not dones[idx].item():
                q_target += self.gamma * torch.max(next_pred[idx]).item()

            target[idx, idx1] = q_target
            target[idx, idx2] = q_target

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        """Sao chép trọng số từ model sang target model."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def get_action(self, board_state, env_bfs_fn):
        """
        Cơ chế Epsilon-Greedy CÓ Action Masking: chỉ xét những cặp ô thực sự
        nối được theo BFS thật của game (luật chơi được lập trình cứng), AI chỉ
        cần học cách chọn cặp nào tốt nhất trong số các nước đi hợp lệ đó.
        - board_state: Ma trận game hiện tại
        - env_bfs_fn: Hàm check bước đi BFS của game
        """
        # Giảm dần epsilon qua từng ván game để AI bớt đi bừa và thông minh dần lên.
        self.epsilon = max(0.01, 80 - self.n_games) / 80

        valid_actions = get_valid_actions(board_state, env_bfs_fn)
        if not valid_actions:
            return None

        # Exploration: thử một cặp hợp lệ ngẫu nhiên.
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation: chọn cặp hợp lệ mà model tin là tốt nhất.
        s_dict = extract_state(board_state)
        s_tensor = state_to_torch(s_dict, device=self.device)["board_onehot"].unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(s_tensor)
        self.model.train()

        q_flat = q_values[0].reshape(-1)
        H, W = len(board_state), len(board_state[0])
        best_action = qvalues_to_action(q_flat, valid_actions, board_width=W, board_height=H)

        return best_action if best_action is not None else valid_actions[0]

if __name__ == '__main__':
    print("--- KHỞI TẠO TEST MÔI TRƯỜNG AGENT OFFLINE ---")
    
    # Tạo một Model CNN giả lập (Dummy Network) thay cho Người số 4 để test lỗi cú pháp
    class DummyCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(101, 1, kernel_size=3, padding=1) # 101 channels khớp max_id=100 của Người 3
            self.fc = torch.nn.Linear(14 * 9, 1) # Bản đồ phẳng phẳng
        def forward(self, x):
            # Giả lập xuất ra một mảng điểm số đơn giản
            return torch.zeros((x.size(0), 1))

    # Cấu hình thử các tham số huấn luyện của PyTorch
    dummy_model = DummyCNN()
    opt = torch.optim.Adam(dummy_model.parameters(), lr=LR)
    crit = torch.nn.MSELoss()
    
    # Khởi tạo bộ não Agent của bạn
    agent = PikachuAgent(model=dummy_model, optimizer=opt, criterion=crit)
    
    # Tạo một board mẫu để kiểm thử tính năng lưu nhớ và học tập
    sample_board = [[0]*14 for _ in range(9)]
    sample_board[1][1] = 1; sample_board[1][2] = 1 # Tạo 1 cặp pokemon loại 1
    
    # Thử nghiệm tính năng Agent lưu và học một bước đi
    print("Thử nghiệm hàm remember và train_short_memory...")
    agent.remember(sample_board, (1, 1, 1, 2), 20, sample_board, False)
    agent.train_short_memory(sample_board, (1, 1, 1, 2), 20, sample_board, False)
    
    print("Thử nghiệm hàm train_long_memory...")
    agent.train_long_memory()