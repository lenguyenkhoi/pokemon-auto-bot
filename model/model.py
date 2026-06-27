"""
model.py - Kien truc Neural Network cho Pokemon AI Agent (DQN).

Input  : board state tensor (batch, channels, height, width)
Output : Q-values tensor (batch, num_actions)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "leaky_relu": F.leaky_relu,
}


class PokemonModel(nn.Module):
    """
    CNN du doan Q-value cho moi action tu board state.
    config: dict, lay tu configs/agent_config.yaml, gom 2 phan can dung:
      - config["model"]: input_channels, hidden_dim, dropout, activation
      - config["environment"]: board_rows, board_cols
      - output_size: so action, truyen rieng vi chua co trong yaml
    """

    def __init__(self, config, output_size):
        super().__init__()
        self.config = config
        m_cfg = config["model"]
        e_cfg = config["environment"]

        self.act_fn = ACTIVATIONS.get(m_cfg.get("activation", "relu"), F.relu)

        in_ch = m_cfg["input_channels"]
        hidden_dim = m_cfg["hidden_dim"]
        dropout = m_cfg["dropout"]
        board_h = e_cfg["board_rows"]
        board_w = e_cfg["board_cols"]

        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        flat_h, flat_w = board_h // 2, board_w // 2
        flatten_size = 64 * flat_h * flat_w

        self.fc1 = nn.Linear(flatten_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, state):
        """Forward pass: state -> Q-values."""
        x = self.act_fn(self.conv1(state))
        x = self.act_fn(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        return self.fc2(x)

    def save(self, path: str):
        """Luu weights model."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        """Load weights model."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()


class QTrainer:
    """Huan luyen PokemonModel theo Bellman equation (DQN)."""

    def __init__(self, model, lr, gamma, target_model=None):
        self.model = model
        self.gamma = gamma
        self.target_model = target_model if target_model is not None else model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        pred = self.model(state)
        target = pred.clone().detach()

        with torch.no_grad():
            next_q = self.target_model(next_state).max(dim=1)[0]

        for i in range(len(done)):
            q_new = reward[i] if done[i] else reward[i] + self.gamma * next_q[i]
            action_idx = action[i].argmax().item() if action[i].dim() > 0 else action[i].item()
            target[i][action_idx] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        """Goi moi target_update_freq steps (config) neu dung DDQN."""
        if self.target_model is not self.model:
            self.target_model.load_state_dict(self.model.state_dict())
