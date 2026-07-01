import torch
import random
import numpy as np
from collections import deque

# Import các hằng số và hàm xử lý Action Masking từ Người số 3 gửi
from utils.state_utils import (
    get_valid_actions, 
    extract_state, 
    state_to_torch, 
    batch_states_to_torch
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
        
        # Thiết bị phần cứng
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def remember(self, state, action, reward, next_state, done):
        """Lưu trữ trải nghiệm (S, A, R, S', done) vào bộ nhớ để học sau"""
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """Học ngay lập tức sau mỗi lượt bấm (Single step training)"""
        # Trích xuất state thô thành dict định dạng numpy chuẩn của Người 3
        s_dict = extract_state(state)
        next_s_dict = extract_state(next_state)
        
        # Chuyển đổi thành Torch Tensor thông qua hàm của Người 3
        s_tensors = state_to_torch(s_dict, device=self.device)
        next_s_tensors = state_to_torch(next_s_dict, device=self.device)
        
        # Tiến hành tối ưu hóa 1 bước đi đơn lẻ
        self._train_step([s_tensors], [action], [reward], [next_s_tensors], [done])

    def train_long_memory(self):
        """Học từ quá khứ sau khi kết thúc một ván game bằng cách bốc ngẫu nhiên một Batch"""
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Bốc ngẫu nhiên một nhóm mẫu từ Replay Memory
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        
        # Phân tách dữ liệu từ mini_batch
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        # Sử dụng hàm batch_states_to_torch rất mạnh của Người 3 để gom cụm tensor hàng loạt
        s_dicts = [extract_state(s) for s in states]
        next_s_dicts = [extract_state(ns) for ns in next_states]
        
        s_tensors = batch_states_to_torch(s_dicts, device=self.device)
        next_s_tensors = batch_states_to_torch(next_s_dicts, device=self.device)
        
        # Tiến hành huấn luyện trên tập dữ liệu lớn
        self._train_step(s_tensors, actions, rewards, next_s_tensors, dones, is_batch=True)

    def _train_step(self, states, actions, rewards, next_states, dones, is_batch=False):
        """Hàm logic tính toán Loss và cập nhật trọng số cho Model của Người 4"""
        # Nếu là batch lớn, lấy thẳng tensor gộp từ Người 3, ngược lại lấy phần tử đơn lẻ
        if is_batch:
            state_input = states["board_onehot"].to(self.device)  # Đầu vào ma trận dạng One-Hot cho CNN
            next_state_input = next_states["board_onehot"].to(self.device)
        else:
            state_input = states[0]["board_onehot"].unsqueeze(0).to(self.device)
            next_state_input = next_states[0]["board_onehot"].unsqueeze(0).to(self.device)
            
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # 1. Dự đoán giá trị Q hiện tại từ model của Người 4
        pred = self.model(state_input)
        target = pred.clone()
        
        # 2. Tính toán giá trị Q mục tiêu dựa trên công thức Bellman: Q_new = R + gamma * max(Q(S', A'))
        with torch.no_grad():
            next_pred = self.model(next_state_input)
            
        H, W = state_input.shape[2], state_input.shape[3]
        for idx in range(len(dones)):
            r1, c1, r2, c2 = actions[idx]
            idx1 = r1 * W + c1
            idx2 = r2 * W + c2
            
            Q_new = rewards[idx]
            if not dones[idx]:
                # Ước lượng max Q(S', A') bằng cách nhân đôi giá trị max cell Q-value
                Q_new = rewards[idx] + self.gamma * torch.max(next_pred[idx]) * 2.0
                
            # Cập nhật mục tiêu học tập cho mạng nơ-ron: chia đều phần chênh lệch cho cả 2 ô của cặp hành động
            current_q = pred[idx][idx1] + pred[idx][idx2]
            diff = Q_new - current_q
            target[idx][idx1] = pred[idx][idx1] + 0.5 * diff
            target[idx][idx2] = pred[idx][idx2] + 0.5 * diff
            
        # 3. Tính toán độ lỗi (Loss) và tối ưu hóa trọng số
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def get_action(self, board_state, env_bfs_fn):
        """
        Cơ chế Epsilon-Greedy: AI quyết định đi bừa hay dùng suy luận từ model.
        - board_state: Ma trận game hiện tại
        - env_bfs_fn: Hàm check bước đi BFS của Người số 2 truyền sang
        """
        # Sử dụng hàm tìm nước đi hợp lệ chuẩn 100% của Người số 3
        valid_actions = get_valid_actions(board_state, env_bfs_fn)
        
        if not valid_actions:
            return None  # Không còn nước đi, trả về None để Env tự động reset

        # Giảm dần epsilon qua từng ván game để AI bớt đi bừa và thông minh dần lên
        self.epsilon = max(0.01, 80 - self.n_games) / 80
        
        # Hành động Khám phá (Exploration): Đi ngẫu nhiên một nước đi đúng luật
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        # Hành động Khai thác (Exploitation): Nhìn ma trận và dùng Model suy luận
        else:
            s_dict = extract_state(board_state)
            s_tensor = state_to_torch(s_dict, device=self.device)["board_onehot"].unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(s_tensor)
            self.model.train()
            
            # Phối hợp với Người 3: Bốc ra action có điểm dự đoán tốt nhất nằm trong danh sách valid
            from utils.state_utils import qvalues_to_action
            best_action = qvalues_to_action(q_values[0], valid_actions, len(board_state[0]), len(board_state))
            if best_action is not None:
                return best_action
            return random.choice(valid_actions)

# ==============================================================================
# ĐOẠN MÃ GIẢ LẬP ĐỂ BẠN TEST FILE AGENT MỘT MÌNH (KHÔNG CẦN NGƯỜI 2 VÀ NGƯỜI 4)
# ==============================================================================
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
    print("--- AGENT ĐÃ SẴN SÀNG CHỜ RÁP NỐI VỚI CẢ NHÓM! ---")