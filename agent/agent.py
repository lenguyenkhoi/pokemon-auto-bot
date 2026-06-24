"""
agent.py - AI Agent chính điều khiển game Pokemon.

Đây là file entry point của Agent. Agent nhận trạng thái game từ Environment,
đưa vào Model để ra quyết định, và gửi action trở lại Environment.

Nhiệm vụ của thành viên phụ trách module này:
  - Implement vòng lặp agent-environment
  - Implement chiến lược khám phá (epsilon-greedy, UCB, ...)
  - Implement replay buffer (nếu dùng RL offline)
  - Lưu / tải checkpoint

Cách dùng:
    python agent.py --config configs/agent_config.yaml --mode play
    python agent.py --config configs/agent_config.yaml --mode train
"""

# TODO: Implement Agent class


class PokemonAgent:
    """
    Agent điều khiển tự động game Pikachu Matching.

    Attributes:
        model: Neural network dự đoán action
        env: Environment game
        config: Cấu hình từ file yaml
    """

    def __init__(self, model, env, config):
        self.model = model
        self.env = env
        self.config = config

    def select_action(self, state):
        """Chọn action dựa trên state hiện tại."""
        raise NotImplementedError("TODO: Implement select_action")

    def run_episode(self):
        """Chạy một episode game."""
        raise NotImplementedError("TODO: Implement run_episode")

    def save_checkpoint(self, path: str):
        """Lưu model checkpoint."""
        raise NotImplementedError("TODO: Implement save_checkpoint")

    def load_checkpoint(self, path: str):
        """Tải model từ checkpoint."""
        raise NotImplementedError("TODO: Implement load_checkpoint")
