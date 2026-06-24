"""
env.py - Game Environment wrapper tương thích Gymnasium (OpenAI Gym).

Wrap game Pikachu Matching thành một RL environment chuẩn,
cung cấp interface: reset(), step(), render(), close()

Nhiệm vụ của thành viên phụ trách module này:
  - Kết nối với game engine (Pikachu-Matching-Game/pikachu.py)
  - Định nghĩa observation space (board state)
  - Định nghĩa action space (chọn 2 ô trên bàn cờ)
  - Tính reward function
  - Implement headless mode (không render UI) cho training nhanh

Observation Space:
    - Board matrix: 2D array shape (rows, cols) chứa pokemon ID (0 = ô trống)
    - Ví dụ: numpy array shape (8, 12), dtype int32

Action Space:
    - Discrete: chọn 2 ô (row1, col1, row2, col2)
    - Hoặc encode thành index phẳng: action = r1*cols*rows*cols + c1*rows*cols + r2*cols + c2

Reward:
    - +1.0  : ghép đôi thành công
    - -0.1  : di chuyển không hợp lệ
    - +10.0 : xóa hết bàn cờ (win)
    - -5.0  : hết thời gian (lose)
"""

# TODO: Implement PokemonEnv


class PokemonEnv:
    """
    Gymnasium-compatible environment cho Pikachu Matching Game.

    Methods:
        reset() -> observation
        step(action) -> (observation, reward, terminated, truncated, info)
        render() -> None
        close() -> None
    """

    def __init__(self, config, headless: bool = True):
        self.config = config
        self.headless = headless

    def reset(self, seed=None):
        """Reset game về trạng thái ban đầu."""
        raise NotImplementedError("TODO: Implement reset()")

    def step(self, action):
        """
        Thực hiện action và trả về (obs, reward, terminated, truncated, info).

        Args:
            action: Action của agent (int hoặc tuple)

        Returns:
            obs: Board state sau action
            reward (float): Phần thưởng nhận được
            terminated (bool): True nếu game kết thúc
            truncated (bool): True nếu hết thời gian
            info (dict): Thông tin thêm (score, combo, ...)
        """
        raise NotImplementedError("TODO: Implement step()")

    def render(self):
        """Render game ra màn hình (chỉ dùng khi demo, không train)."""
        raise NotImplementedError("TODO: Implement render()")

    def close(self):
        """Giải phóng tài nguyên."""
        raise NotImplementedError("TODO: Implement close()")
