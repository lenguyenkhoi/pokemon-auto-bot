"""
model.py - Kiến trúc Neural Network cho Pokemon AI Agent.

File này định nghĩa model deep learning được dùng để dự đoán
action (Q-values hoặc policy distribution) từ trạng thái game.

Nhiệm vụ của thành viên phụ trách module này:
  - Thiết kế kiến trúc CNN/DNN nhận input là board state
  - Implement forward pass
  - Implement hàm loss nếu cần custom loss
  - Đảm bảo model export/import được (state_dict)

Framework gợi ý: PyTorch

Input shape:
  - Board state: (batch, channels, height, width)
  - Ví dụ board 8x12: tensor shape (B, 1, 8, 12) hoặc one-hot (B, N_pokemon, 8, 12)

Output:
  - Q-values: tensor shape (B, num_actions)  -- nếu dùng DQN
  - Policy logits + value: -- nếu dùng Actor-Critic / PPO
"""

# TODO: Implement Model class


class PokemonModel:
    """
    Neural Network dự đoán action cho Pokemon Matching game.

    Có thể implement theo một trong các hướng:
      - DQN  : output Q(s,a) cho mỗi action
      - PPO  : output policy π(a|s) + value V(s)
      - A2C  : tương tự PPO nhưng on-policy đơn giản hơn
    """

    def __init__(self, config):
        self.config = config

    def forward(self, state):
        """Forward pass: state -> action logits / Q-values."""
        raise NotImplementedError("TODO: Implement forward pass")

    def save(self, path: str):
        """Lưu weights model."""
        raise NotImplementedError("TODO: Implement save")

    def load(self, path: str):
        """Load weights model."""
        raise NotImplementedError("TODO: Implement load")
