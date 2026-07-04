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
from pikachu import PikachuEnv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    config_path = os.path.join(ROOT_DIR, "configs", "agent_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plot_live(scores, mean_scores, epsilons, fig, ax1, ax2):
    """Cap nhat bieu do truc quan thoi gian thuc 
    (Live Plotting)"""
    ax1.clear()
    ax2.clear()
    
    # Ve diem so va diem trung binh
    ax1.set_title("Qua trinh Huan luyen - Diem so (Scores)")
    ax1.set_xlabel("So van (Episodes)")
    ax1.set_ylabel("Diem (Score)")
    ax1.plot(scores, label="Diem tung van", color="skyblue", alpha=0.6)
    ax1.plot(mean_scores, label="Trung binh 10 van", color="blue", linewidth=2)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # Ve epsilon decay
    ax2.set_title("Toc do giam Epsilon (Epsilon Decay)")
    ax2.set_xlabel("So van (Episodes)")
    ax2.set_ylabel("Epsilon")
    ax2.plot(epsilons, label="Epsilon", color="red", linewidth=2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.05)  # Tam dung ngan de matplotlib cap nhat UI

def train():
    print("=== HUAN LUYEN TIEP TUC MODEL 3 CHO POKEMON AUTO BOT ===")
    
    # Kiem tra thiet bi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiet bi huan luyen dang dung: {device.upper()}")
    
    config = load_config()
    
    # Sieu tham so cho huan luyen tiep tuc
    EPISODES = 150  # So vao choi huan luyen thuc te
    LR = 0.0005     # Learning rate on dinh
    GAMMA = 0.95    # Quan tam hon den nuoc di tuong lai
    
    # Khoi tao Moi truong
    board_width = config["environment"]["board_cols"]
    board_height = config["environment"]["board_rows"]
    # Khoi tao Moi truong (bat render_mode=True de xem AI choi trong luc train)
    env = PikachuEnv(board_width=board_width, board_height=board_height, level=1, render_mode=True)
    
    # Output size = H * W
    output_size = board_width * board_height
    
    # Khoi tao Model CNN
    model = PokemonModel(config, output_size).to(device)
    
    # Nap checkpoint de train tiep tu file cu neu ton tai
    model_path = os.path.join(ROOT_DIR, "weights", "best_model_3.pt")
    initial_n_games = 0
    if os.path.exists(model_path):
        print(f"--> Phat hien checkpoint cu (Model 3) tai: {model_path}")
        print("Dang nap trong so cu de tiep tuc huan luyen...")
        model.load(model_path, map_location=device)
        # Tiep tuc epsilon da giam
        initial_n_games = 45
    else:
        print("--> Khong tim thay checkpoint truoc do. Bat dau huan luyen lai tu dau.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    
    # Khoi tao Agent voi device truyen vao
    agent = PikachuAgent(model, optimizer, criterion, device=device)
    agent.gamma = GAMMA
    agent.n_games = initial_n_games
    
    scores = []
    mean_scores = []
    epsilons = []
    
    best_score = -float('inf')
    
    # Kich hoat che do ve do thi truc tiep (Interactive Mode)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    import pygame  # Import pygame de xu ly su kien
    
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Xu ly su kien Pygame de cua so khong bi treo (Not Responding)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    plt.close()
                    sys.exit()
                    
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
        
        # Cap nhat do thi truc tiep sau moi episode
        plot_live(scores, mean_scores, epsilons, fig, ax1, ax2)
        
        # Luu best checkpoint khi model dat diem cao hon
        if env.score > best_score:
            best_score = env.score
            weights_dir = os.path.join(ROOT_DIR, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            model.save(os.path.join(weights_dir, "best_model_3.pt"))
            print(f"--> Da luu best model moi tai weights/best_model.pt voi Score: {best_score}")
            
    # Tat che do ve truc tiep, luu do thi cuoi cung
    plt.ioff()
    save_path = os.path.join(ROOT_DIR, "training_progress.png")
    plt.savefig(save_path)
    print(f"Da luu bieu do huan luyen cuoi cung tai: {save_path}")
    plt.show()
    print("=== HUAN LUYEN HOAN THANH! ===")

if __name__ == "__main__":
    train()
