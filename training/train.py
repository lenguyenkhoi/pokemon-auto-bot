"""
train.py - Vòng lặp huấn luyện AI Agent Pokemon.

File này orchestrate toàn bộ quá trình training:
  Environment → Agent → Model → Optimizer → Checkpoint

Nhiệm vụ của thành viên phụ trách module này:
  - Implement training loop (episodes, steps)
  - Implement evaluation loop (no gradient)
  - Log metrics (reward, loss, win rate) ra TensorBoard / wandb
  - Lưu checkpoint định kỳ
  - Early stopping nếu cần

Cách chạy:
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --resume logs/checkpoints/ep_100.pt

Output:
    - Model checkpoints tại: logs/checkpoints/
    - TensorBoard logs tại:  logs/tensorboard/
    - Training runs tại:     logs/runs/
"""

# TODO: Implement training pipeline


def train(config):
    """
    Vòng lặp huấn luyện chính.

    Args:
        config (dict): Cấu hình training (lr, batch_size, n_episodes, ...)
    """
    raise NotImplementedError("TODO: Implement train()")


def evaluate(agent, env, n_episodes: int = 10):
    """
    Đánh giá agent trên n_episodes game (không update weights).

    Args:
        agent: Agent đã được train
        env: Environment game
        n_episodes: Số episode đánh giá

    Returns:
        dict: Metrics (avg_reward, win_rate, avg_steps)
    """
    raise NotImplementedError("TODO: Implement evaluate()")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Pokemon AI Agent")
    parser.add_argument("--config", type=str, required=True, help="Path to train config yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    # TODO: Load config và gọi train()
    print(f"Training with config: {args.config}")
