"""
main.py - Entry point chính để chạy Pokemon AI Agent.

Usage:
    # Chạy AI tự động chơi game (demo mode)
    python main.py --mode play --checkpoint logs/checkpoints/best.pt

    # Huấn luyện agent mới
    python main.py --mode train --config configs/train_config.yaml

    # Đánh giá agent
    python main.py --mode eval --checkpoint logs/checkpoints/best.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Pokemon AI Agent")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["play", "train", "eval"],
        default="play",
        help="Chế độ chạy: play (demo) | train (huấn luyện) | eval (đánh giá)"
    )
    parser.add_argument("--config", type=str, default="configs/agent_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        from training.train import train
        train(args.config)
    elif args.mode == "play":
        print("TODO: Implement play mode")
    elif args.mode == "eval":
        print("TODO: Implement eval mode")


if __name__ == "__main__":
    main()
