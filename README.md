# Pokemon Auto-Bot

A Deep Q-Learning agent that learns to play the [Pikachu Matching Game](Pikachu-Matching-Game/README.md) — an Onet/Pikachu-style tile-matching game built with Pygame — and can then take over and play the real game live, driven by mouse clicks.

## How it works

| Piece | File | Role |
|---|---|---|
| Environment | `Pikachu-Matching-Game/pikachu.py` | Exposes `PikachuEnv` (`reset()` / `play_step(action)`), wrapping the actual game so it can be driven programmatically. The board is a `9 x 14` grid (`configs/agent_config.yaml`); each cell is `0` (empty) or a Pokémon type ID. |
| State | `utils/state_utils.py` | Converts the raw board into one-hot tensors for the CNN, finds every currently valid move via BFS (mirroring the game's own connect-with-at-most-2-turns rule), and applies reward shaping. |
| Model | `model/model.py` | `PokemonModel`, a small CNN that takes the one-hot board and predicts a Q-value per cell; `QTrainer` implements the Bellman update against a target network. |
| Agent | `agent/agent.py` | `PikachuAgent` — epsilon-greedy action selection **with action masking**, so only legal pairs (per BFS) are ever considered. Keeps a replay buffer and trains on both single steps (short memory) and sampled batches (long memory). |
| Training | `train.py` | Runs episodes against the real environment, live-plots score/epsilon with matplotlib, and checkpoints the best-scoring model to `weights/best_model.pt`. |
| Demo | `demo.py` | Launches the game with `--ai`, which loads the trained weights and lets the agent play live in the Pygame window. |

## Project structure

```
pokemon-auto-bot/
├── agent/                  # PikachuAgent: epsilon-greedy DQN with action masking + replay memory
├── model/                  # PokemonModel (CNN) and QTrainer (Bellman update)
├── utils/                  # Board <-> tensor conversion, BFS valid-move search, reward shaping
├── configs/
│   └── agent_config.yaml   # Model hyperparameters + board dimensions
├── weights/                # Saved model checkpoints (best_model*.pt)
├── tests/                  # Unit tests for utils/state_utils.py
├── test_model/             # Unit tests for model/model.py
├── Pikachu-Matching-Game/  # The Pygame matching game itself (playable standalone or by the AI)
├── train.py                 # Training loop
├── demo.py                  # Runs the game with the trained AI playing live
└── training_progress.png    # Score/epsilon chart from the last training run
```

## Setup

Requires Python >= 3.8.

```bash
pip install -r Pikachu-Matching-Game/requirements.txt
```

## Usage

### Watch the trained AI play

```bash
python demo.py
```

Starts the game with `--ai`, which loads `weights/best_model.pt` and lets the agent play automatically.

### Train (or continue training) the agent

```bash
python train.py
```

Loads `weights/best_model.pt` as a starting checkpoint if it exists, otherwise trains from scratch. Live score/epsilon plots are shown during training and saved to `training_progress.png` at the end; the best-scoring checkpoint is saved back to `weights/best_model.pt`.

### Play the game yourself

```bash
python Pikachu-Matching-Game/pikachu.py
```

### Run tests

```bash
pytest tests/ test_model/
```

## Model details

- **Input**: one-hot encoded board, `101` channels (one per Pokémon ID + empty).
- **Architecture**: two `Conv2d` layers (32 then 64 filters) → max pool → dropout → flatten → fully-connected hidden layer → output layer.
- **Output**: one Q-value per board cell (`9 x 14 = 126` values); the best legal pair is chosen by summing the Q-values of its two cells (`utils/state_utils.qvalues_to_action`).
- **Algorithm**: DQN with a target network, epsilon-greedy exploration decayed over episodes, and shaped rewards (bonus for clearing a pair, larger bonus for clearing the board, penalty for invalid/timeout moves).
