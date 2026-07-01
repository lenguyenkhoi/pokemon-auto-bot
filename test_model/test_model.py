import yaml
import torch
from model import PokemonModel, QTrainer


def load_config():
    with open("configs/agent_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_forward_shape():
    config = load_config()
    output_size = 30

    model = PokemonModel(config, output_size)

    c = config["model"]["input_channels"]
    h = config["environment"]["board_rows"]
    w = config["environment"]["board_cols"]

    dummy_input = torch.zeros(4, c, h, w)
    output = model(dummy_input)

    assert output.shape == (4, output_size)
    print("Forward pass OK, output shape:", output.shape)


def test_train_step_runs():
    config = load_config()
    output_size = 30

    model = PokemonModel(config, output_size)
    trainer = QTrainer(model, lr=0.001, gamma=0.99)

    c = config["model"]["input_channels"]
    h = config["environment"]["board_rows"]
    w = config["environment"]["board_cols"]
    batch = 4

    state = torch.rand(batch, c, h, w)
    next_state = torch.rand(batch, c, h, w)
    action = torch.zeros(batch, output_size)
    for i in range(batch):
        action[i, i % output_size] = 1
    reward = torch.tensor([1.0, -1.0, 0.0, 10.0])
    done = torch.tensor([False, False, False, True])

    loss = trainer.train_step(state, action, reward, next_state, done)
    print("Loss:", loss)
    assert isinstance(loss, float)


def test_save_and_load(tmp_path):
    config = load_config()
    output_size = 30
    model = PokemonModel(config, output_size)
    model.eval()  # tat dropout truoc khi so sanh, tranh ket qua bi random

    save_path = tmp_path / "test_model.pt"
    model.save(str(save_path))

    model2 = PokemonModel(config, output_size)
    model2.load(str(save_path))  # load() da tu goi eval() ben trong

    c = config["model"]["input_channels"]
    h = config["environment"]["board_rows"]
    w = config["environment"]["board_cols"]
    x = torch.rand(1, c, h, w)

    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2)
