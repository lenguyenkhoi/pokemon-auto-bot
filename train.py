import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.model import PokemonModel
from agent.agent import PikachuAgent

# Them thu muc game vao sys.path de import vi ten thu muc co dau gach ngang
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pikachu-Matching-Game"))
# pyrefly: ignore [missing-import]
from pikachu_env import PikachuEnv

def load_config():
    with open("configs/agent_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plot_progress(scores, mean_scores, epsilons, save_path="training_progress.png"):
    plt.figure(figsize=(12, 6))
    
    # Ve diem so va diem trung binh
    plt.subplot(1, 2, 1)
    plt.title("Qua trinh Huan luyen - Diem so (Scores)")
    plt.xlabel("So van (Episodes)")
    plt.ylabel("Diem (Score)")
    plt.plot(scores, label="Diem tung van", color="skyblue", alpha=0.6)
    plt.plot(mean_scores, label="Trung binh 10 van", color="blue", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    # Ve epsilon decay
    plt.subplot(1, 2, 2)
    plt.title("Toc do giam Epsilon (Epsilon Decay)")
    plt.xlabel("So van (Episodes)")
    plt.ylabel("Epsilon")
    plt.plot(epsilons, label="Epsilon", color="red", linewidth=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Da luu bieu do huan luyen tai: {save_path}")

def train():
    print("=== HUAN LUYEN MODEL THAT CHO POKEMON AUTO BOT ===")
    
    # Kiem tra thiet bi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiet bi huan luyen dang dung: {device.upper()}")
    
    config = load_config()
    
    # Sieu tham so cho huan luyen that
    EPISODES = 150  # So vao choi huan luyen thuc te
    LR = 0.0005     # Learning rate on dinh
    GAMMA = 0.95    # Quan tam hon den nuoc di tuong lai
    
    # Khoi tao Moi truong
    board_width = config["environment"]["board_cols"]
    board_height = config["environment"]["board_rows"]
    env = PikachuEnv(board_width=board_width, board_height=board_height, level=1, render_mode=False)
    
    # Output size = H * W
    output_size = board_width * board_height
    
    # Khoi tao Model CNN
    model = PokemonModel(config, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    
    # Khoi tao Agent voi device truyen vao
    agent = PikachuAgent(model, optimizer, criterion, device=device)
    agent.gamma = GAMMA
    
    scores = []
    mean_scores = []
    epsilons = []
    
    best_score = -float('inf')
    
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-Greedy action selection
            action = agent.get_action(state, env.bfs)
            
            if action is None:
                break
                
            # Step environment
            next_state, reward, done, score = env.play_step(action)
            total_reward += reward
            
            # Remember transition
            agent.remember(state, action, reward, next_state, done)
            
            # Short term memory training
            agent.train_short_memory(state, action, reward, next_state, done)
            
            state = next_state
            
        # Long term memory training
        agent.train_long_memory()
        
        # Tang dem game de cap nhat epsilon decay
        agent.n_games += 1
        
        # Luu tru ket qua
        scores.append(env.score)
        mean_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        mean_scores.append(mean_score)
        epsilons.append(agent.epsilon)
        
        # Log tien do
        print(f"Episode {episode:03d}/{EPISODES} | Score: {env.score:03d} | Mean Score (10 eps): {mean_score:.2f} | Epsilon: {agent.epsilon:.3f} | Total Reward: {total_reward:.1f}")
        
        # Luu best checkpoint khi model dat diem cao hon
        if env.score > best_score:
            best_score = env.score
            os.makedirs("weights", exist_ok=True)
            model.save("weights/best_model.pt")
            print(f"--> Da luu best model moi tai weights/best_model.pt voi Score: {best_score}")
            
    # Ve va luu bieu do
    plot_progress(scores, mean_scores, epsilons, save_path="training_progress.png")
    print("=== HUAN LUYEN HOAN THANH! ===")

if __name__ == "__main__":
    train()
