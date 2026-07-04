# -*- coding: utf-8 -*-
"""
tests/test_env/test_state_utils.py
===================================
Unit test cho utils/state_utils.py — thuộc phạm vi Người 3.

Đặt trong tests/test_env/ vì state_utils phục vụ trực tiếp
cho environment (Người 2) và là cầu nối game ↔ agent.

Chạy từ thư mục gốc pokemon-auto-bot/GAME/:
    pytest tests/test_env/test_state_utils.py -v

Hoặc toàn bộ test suite:
    pytest tests/ -v
"""

import sys
import os

# Thêm root project vào sys.path để import utils
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
import torch
from agent.agent import PikachuAgent
from utils.state_utils import (
    board_to_numpy, board_to_onehot, board_to_tensor,
    board_to_flat_tensor, normalize_board, get_state_info,
    get_valid_actions, get_one_valid_action,
    action_to_pair_index, pair_index_to_action,
    encode_action, decode_action, build_action_mask,
    count_remaining, is_board_empty, board_progress,
)


# ─── Fixture: board mẫu 5×6 (H=5, W=6) ──────────────────────────────────────
#  Viền ngoài = 0 (buffer BFS), vùng trong có pokemon
#
#  col:  0  1  2  3  4  5
#  row0: 0  0  0  0  0  0   ← viền trên
#  row1: 0  1  2  1  0  0   ← (1,1)=pk1, (1,2)=pk2, (1,3)=pk1
#  row2: 0  3  0  3  0  0   ← (2,1)=pk3, (2,3)=pk3
#  row3: 0  2  0  0  0  0   ← (3,1)=pk2
#  row4: 0  0  0  0  0  0   ← viền dưới
SAMPLE_BOARD = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0, 0],
    [0, 3, 0, 3, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

EMPTY_BOARD = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]


# BFS giả lập: cùng loại + (cùng hàng HOẶC cùng cột) → nối được
# Đủ để test logic của state_utils mà không cần import pygame/pikachu
def fake_bfs(board, r1, c1, r2, c2):
    if board[r1][c1] == 0 or board[r1][c1] != board[r2][c2]:
        return []
    if r1 == r2 or c1 == c2:
        return [(r1, c1), (r2, c2)]
    return []


# ══════════════════════════════════════════════════════════════════════════════
class TestBoardConversions:
    """Kiểm tra các hàm chuyển đổi board → tensor (Phần 1)."""

    def test_board_to_numpy_shape_and_dtype(self):
        arr = board_to_numpy(SAMPLE_BOARD)
        assert arr.shape == (5, 6)
        assert arr.dtype == np.int32

    def test_board_to_numpy_values_correct(self):
        arr = board_to_numpy(SAMPLE_BOARD)
        assert arr[1][1] == 1    # pokemon loại 1
        assert arr[1][2] == 2    # pokemon loại 2
        assert arr[0][0] == 0    # viền ngoài = 0

    def test_board_to_onehot_shape(self):
        onehot = board_to_onehot(SAMPLE_BOARD, num_classes=4)
        assert onehot.shape == (4, 5, 6)   # (C, H, W)
        assert onehot.dtype == np.float32

    def test_board_to_onehot_channel0_is_empty_cells(self):
        onehot = board_to_onehot(SAMPLE_BOARD, num_classes=4)
        assert onehot[0][0][0] == 1.0    # (0,0) là viền trống → ch0=1
        assert onehot[0][1][1] == 0.0    # (1,1) có pokemon → ch0=0

    def test_board_to_onehot_correct_pokemon_channel(self):
        onehot = board_to_onehot(SAMPLE_BOARD, num_classes=4)
        assert onehot[1][1][1] == 1.0    # board[1][1]=1 → channel 1
        assert onehot[2][1][2] == 1.0    # board[1][2]=2 → channel 2
        assert onehot[3][2][1] == 1.0    # board[2][1]=3 → channel 3

    def test_board_to_onehot_sum_per_cell_equals_one(self):
        """Mỗi ô phải thuộc đúng 1 class (tổng channel = 1.0)."""
        onehot = board_to_onehot(SAMPLE_BOARD, num_classes=4)
        channel_sum = onehot.sum(axis=0)   # (H, W)
        assert np.allclose(channel_sum, 1.0)

    def test_board_to_tensor_shape(self):
        t = board_to_tensor(SAMPLE_BOARD, num_classes=4)
        assert tuple(t.shape) == (1, 4, 5, 6)

    def test_board_to_flat_tensor_shape(self):
        t = board_to_flat_tensor(SAMPLE_BOARD)
        assert tuple(t.shape) == (1, 30)   # 5×6=30

    def test_normalize_board_range(self):
        arr = normalize_board(SAMPLE_BOARD, max_type=3)
        assert arr.dtype == np.float32
        assert float(arr.max()) <= 1.0
        assert float(arr.min()) >= 0.0

    def test_normalize_board_zero_max_no_crash(self):
        """max_type=0 không được gây ZeroDivisionError."""
        arr = normalize_board(EMPTY_BOARD, max_type=0)
        assert arr is not None


class TestGetStateInfo:
    """Kiểm tra metadata của board (Phần 1 — get_state_info)."""

    def test_count_pokemon_on_sample_board(self):
        info = get_state_info(SAMPLE_BOARD)
        # (1,1),(1,2),(1,3),(2,1),(2,3),(3,1) → 6 ô
        assert info["num_remaining"] == 6

    def test_num_types_on_sample_board(self):
        info = get_state_info(SAMPLE_BOARD)
        assert info["num_types"] == 3
        assert set(info["unique_types"]) == {1, 2, 3}

    def test_board_shape_in_info(self):
        info = get_state_info(SAMPLE_BOARD)
        assert info["board_shape"] == (5, 6)

    def test_empty_board_state_info(self):
        info = get_state_info(EMPTY_BOARD)
        assert info["num_remaining"] == 0
        assert info["num_types"] == 0
        assert info["unique_types"] == []


class TestGetValidActions:
    """Kiểm tra get_valid_actions — hàm cốt lõi của Người 3 (Phần 2)."""

    def test_returns_list(self):
        actions = get_valid_actions(SAMPLE_BOARD, fake_bfs)
        assert isinstance(actions, list)

    def test_no_duplicate_pairs(self):
        """Mỗi cặp ô chỉ xuất hiện đúng 1 lần."""
        actions = get_valid_actions(SAMPLE_BOARD, fake_bfs)
        assert len(actions) == len(set(actions))

    def test_all_pairs_same_type(self):
        """Chỉ cùng loại pokemon mới được ghép."""
        actions = get_valid_actions(SAMPLE_BOARD, fake_bfs)
        for r1, c1, r2, c2 in actions:
            assert SAMPLE_BOARD[r1][c1] == SAMPLE_BOARD[r2][c2], \
                f"Cặp ({r1},{c1})-({r2},{c2}) khác loại!"

    def test_no_zero_cells_in_pairs(self):
        """Không được chọn ô trống."""
        actions = get_valid_actions(SAMPLE_BOARD, fake_bfs)
        for r1, c1, r2, c2 in actions:
            assert SAMPLE_BOARD[r1][c1] != 0
            assert SAMPLE_BOARD[r2][c2] != 0

    def test_empty_board_returns_empty_list(self):
        actions = get_valid_actions(EMPTY_BOARD, fake_bfs)
        assert actions == []

    def test_ordering_i_less_than_j(self):
        """Đảm bảo không sinh cặp trùng (i<j)."""
        actions = get_valid_actions(SAMPLE_BOARD, fake_bfs)
        for r1, c1, r2, c2 in actions:
            pos1 = (r1, c1)
            pos2 = (r2, c2)
            # Pair (A,B) không xuất hiện cùng lúc với (B,A)
            assert (r2, c2, r1, c1) not in actions, \
                f"Trùng cặp: ({r1},{c1})-({r2},{c2})"

    def test_get_one_valid_action_returns_valid_or_none(self):
        action = get_one_valid_action(SAMPLE_BOARD, fake_bfs)
        if action is not None:
            r1, c1, r2, c2 = action
            assert SAMPLE_BOARD[r1][c1] == SAMPLE_BOARD[r2][c2]

    def test_get_one_valid_action_empty_board(self):
        assert get_one_valid_action(EMPTY_BOARD, fake_bfs) is None


def test_agent_can_choose_non_valid_pair_when_model_prefers_it(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = torch.full((x.size(0), 14 * 9), -1.0)
            out[0, 1 * 14 + 1] = 100.0
            out[0, 1 * 14 + 3] = 100.0
            return out

    board = [[0] * 14 for _ in range(9)]
    board[1][1] = 1
    board[1][2] = 1
    board[1][3] = 2

    agent = PikachuAgent(
        model=DummyModel(),
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
        criterion=torch.nn.MSELoss(),
        device="cpu",
    )
    agent.n_games = 1000
    monkeypatch.setattr("random.random", lambda: 1.0)

    action = agent.get_action(board, lambda *args, **kwargs: [])

    assert action == (1, 1, 1, 3)


class TestActionEncoding:
    """Kiểm tra encode/decode action (Phần 3)."""

    def test_pair_index_roundtrip(self):
        valid = [(1, 1, 1, 3), (2, 1, 2, 3), (1, 2, 3, 1)]
        for i, action in enumerate(valid):
            assert action_to_pair_index(action, valid) == i
            assert pair_index_to_action(i, valid) == action

    def test_pair_index_not_found(self):
        valid = [(1, 1, 1, 3)]
        assert action_to_pair_index((9, 9, 9, 9), valid) == -1

    def test_pair_index_out_of_range(self):
        valid = [(1, 1, 1, 3)]
        assert pair_index_to_action(99, valid) is None
        assert pair_index_to_action(-1, valid) is None

    def test_encode_decode_roundtrip(self):
        action = (1, 1, 1, 3)
        encoded = encode_action(action, board_width=6, board_height=5)
        decoded = decode_action(encoded, board_width=6, board_height=5)
        assert decoded == action

    def test_encode_unique_per_action(self):
        """Mỗi action phải có mã khác nhau."""
        a1 = encode_action((1, 1, 1, 3), 6, 5)
        a2 = encode_action((2, 1, 2, 3), 6, 5)
        a3 = encode_action((1, 1, 2, 1), 6, 5)
        assert len({a1, a2, a3}) == 3

    def test_build_action_mask_shape_and_values(self):
        all_actions = [(1, 1, 1, 3), (2, 1, 2, 3), (0, 0, 0, 1)]
        valid       = [(1, 1, 1, 3)]
        mask = build_action_mask(valid, all_actions)
        assert mask.shape == (3,)
        assert mask[0] == 1.0    # hợp lệ
        assert mask[1] == 0.0    # không hợp lệ
        assert mask[2] == 0.0    # không hợp lệ

    def test_build_action_mask_empty_valid(self):
        all_actions = [(1, 1, 1, 3), (2, 1, 2, 3)]
        mask = build_action_mask([], all_actions)
        assert (mask == 0.0).all()

    def test_build_action_mask_all_valid(self):
        all_actions = [(1, 1, 1, 3), (2, 1, 2, 3)]
        mask = build_action_mask(all_actions, all_actions)
        assert (mask == 1.0).all()


class TestBoardUtils:
    """Kiểm tra các tiện ích đo lường board (Phần 4)."""

    def test_count_remaining_sample(self):
        assert count_remaining(SAMPLE_BOARD) == 6

    def test_count_remaining_empty(self):
        assert count_remaining(EMPTY_BOARD) == 0

    def test_is_board_empty_true(self):
        assert is_board_empty(EMPTY_BOARD) is True

    def test_is_board_empty_false(self):
        assert is_board_empty(SAMPLE_BOARD) is False

    def test_board_progress_no_remaining(self):
        assert board_progress(EMPTY_BOARD, initial_count=10) == 1.0 - 0/10

    def test_board_progress_partial(self):
        # còn 6 ô, ban đầu 10 → đã xóa 4/10 = 40%
        p = board_progress(SAMPLE_BOARD, initial_count=10)
        assert abs(p - 0.4) < 1e-6

    def test_board_progress_zero_initial(self):
        # Không crash khi initial_count=0
        assert board_progress(SAMPLE_BOARD, initial_count=0) == 1.0

    def test_board_progress_full_clear(self):
        assert board_progress(EMPTY_BOARD, initial_count=6) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
