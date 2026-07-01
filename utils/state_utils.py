# -*- coding: utf-8 -*-
"""
utils/state_utils.py  (bản GỘP)
================================
File này gộp 2 module trùng chức năng: state_utils.py + game_interface.py
thành 1 module duy nhất, giữ nguyên toàn bộ API cũ (để test_state_utils.py
chạy được không cần sửa) và bổ sung các hàm nâng cao (extract_state,
torch tensor, reward shaping, apply_action, qvalues_to_action...).

Vị trí đặt file trong project:
    pokemon-auto-bot/GAME/utils/state_utils.py

────────────────────────────────────────────────────────────────────────────
CẤU TRÚC BOARD (khớp pikachu.py)
    board[row][col]
        = 0   → ô trống (viền ngoài luôn = 0, là buffer cho BFS)
        > 0   → ID loại pokemon (1 .. NUMHEROES_ONBOARD)
    Action chuẩn dùng trong file này: tuple phẳng (r1, c1, r2, c2)

────────────────────────────────────────────────────────────────────────────
CÁCH CHẠY KHI ĐÃ CÓ FOLDER GAME (Pikachu-Matching-Game) TRÊN MÁY LOCAL
────────────────────────────────────────────────────────────────────────────
1. Đặt file này vào đúng đường dẫn:
       pokemon-auto-bot/GAME/utils/state_utils.py

2. Đảm bảo utils/__init__.py export đúng các hàm (giữ nguyên như cũ):
       from .state_utils import (
           board_to_numpy, board_to_onehot, board_to_tensor,
           board_to_flat_tensor, normalize_board, get_state_info,
           get_valid_actions, get_one_valid_action,
           action_to_pair_index, pair_index_to_action,
           encode_action, decode_action, build_action_mask,
           count_remaining, is_board_empty, board_progress,
       )

3. Import hàm bfs() THẬT từ game gốc (trong environment/env.py của Người 2):
       import sys, os
       sys.path.insert(0, os.path.join(ROOT, "Pikachu-Matching-Game"))
       from pikachu import bfs

       from utils import get_valid_actions
       valid = get_valid_actions(board, bfs)     # dùng BFS thật của game

   -> Nếu KHÔNG có sẵn hàm bfs() của game (ví dụ train offline, chưa có
      pygame), chỉ cần KHÔNG truyền bfs_fn — file này có sẵn BFS nội bộ
      (không phụ thuộc pygame) sẽ tự động được dùng thay thế:
       valid = get_valid_actions(board)          # dùng BFS nội bộ

4. Chạy unit test (không cần cài pygame, không cần có game thật):
       cd pokemon-auto-bot/GAME
       pytest tests/test_env/test_state_utils.py -v

5. Chạy demo nhanh (kiểm tra toàn bộ pipeline: board -> tensor -> action
   -> reward) ngay trong file này:
       python utils/state_utils.py
"""

from __future__ import annotations

import copy
import collections
import random
from typing import List, Tuple, Optional, Callable, Dict, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ─── Type alias ──────────────────────────────────────────────────────────
Board  = List[List[int]]
Action = Tuple[int, int, int, int]      # (row1, col1, row2, col2)
BfsFn  = Callable[..., list]            # bfs(board, r1, c1, r2, c2) -> path | []

# ─── Hằng số mặc định (khớp pikachu.py) ────────────────────────────────
DEFAULT_BOARD_HEIGHT = 9
DEFAULT_BOARD_WIDTH  = 14
MAX_POKEMON_ID        = 100
MAX_SCORE              = 1000.0
MAX_LIVES              = 10
MAX_TIME                = 360.0
MAX_LEVEL               = 5


# ══════════════════════════════════════════════════════════════════════
# PHẦN 1 — CHUYỂN ĐỔI BOARD -> NUMPY / TENSOR
# ══════════════════════════════════════════════════════════════════════

def board_to_numpy(board: Board) -> np.ndarray:
    """Chuyển board 2D -> numpy array dtype int32. Shape (H, W)."""
    return np.array(board, dtype=np.int32)


def board_to_onehot(board: Board, num_classes: int) -> np.ndarray:
    """
    One-hot hoá board.
    Shape: (num_classes, H, W) float32.
        Channel 0   : ô trống (board == 0)
        Channel k>0 : loại pokemon k
    num_classes = số loại pokemon + 1 (= len(HEROES_DICT) + 1).
    """
    arr = board_to_numpy(board)
    H, W = arr.shape
    onehot = np.zeros((num_classes, H, W), dtype=np.float32)
    for c in range(num_classes):
        onehot[c] = (arr == c).astype(np.float32)
    return onehot


def board_to_tensor(board: Board, num_classes: int, device: str = "cpu") -> "torch.Tensor":
    """Board -> tensor CNN sẵn sàng dùng. Shape (1, num_classes, H, W)."""
    onehot = board_to_onehot(board, num_classes)
    if _TORCH_AVAILABLE:
        tensor = torch.tensor(onehot, dtype=torch.float32)
        return tensor.unsqueeze(0).to(device)
    return np.expand_dims(onehot.astype(np.float32), axis=0)


def board_to_flat_tensor(board: Board, device: str = "cpu") -> "torch.Tensor":
    """Board -> tensor 1D flatten. Shape (1, H*W). Dùng cho model Linear/MLP."""
    arr = board_to_numpy(board).flatten().astype(np.float32)
    if _TORCH_AVAILABLE:
        return torch.tensor(arr).unsqueeze(0).to(device)
    return np.expand_dims(arr, axis=0)


def normalize_board(board: Board, max_type: int) -> np.ndarray:
    """Chuẩn hoá board về [0,1] (chia cho max_type). Shape (H, W) float32."""
    arr = board_to_numpy(board).astype(np.float32)
    if max_type > 0:
        arr /= float(max_type)
    return arr


def get_state_info(board: Board) -> dict:
    """
    Metadata của board hiện tại:
        num_remaining, num_types, board_shape, unique_types
    """
    arr = board_to_numpy(board)
    non_zero = arr[arr > 0]
    return {
        "num_remaining": int(non_zero.size),
        "num_types"    : int(np.unique(non_zero).size) if non_zero.size > 0 else 0,
        "board_shape"  : arr.shape,
        "unique_types" : np.unique(non_zero).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════
# PHẦN 2 — BFS NỘI BỘ (không phụ thuộc pygame)
# Dùng làm fallback khi không truyền bfs_fn từ pikachu.py vào.
# ══════════════════════════════════════════════════════════════════════

def _bfs_check(board: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
    """
    Kiểm tra 2 ô có nối được nhau không (đường đi <= 2 lần rẽ),
    mô phỏng luật game Pikachu, KHÔNG cần import pygame.
    """
    if board[r1, c1] == 0 or board[r2, c2] == 0:
        return False
    if board[r1, c1] != board[r2, c2]:
        return False
    if (r1, c1) == (r2, c2):
        return False

    H, W = board.shape
    DIR_MAP = {(1, 0): 2, (-1, 0): 1, (0, -1): 3, (0, 1): 4}

    q = collections.deque([(r1, c1, 0, 0)])
    visited = {(r1, c1, 0, 0)}

    while q:
        r, c, turns, direction = q.popleft()
        for (dr, dc), ndir in DIR_MAP.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if board[nr, nc] != 0 and (nr, nc) != (r2, c2):
                continue
            new_turns = turns if (direction == 0 or direction == ndir) else turns + 1
            if new_turns > 2:
                continue
            if (nr, nc) == (r2, c2):
                return True
            state = (nr, nc, new_turns, ndir)
            if state not in visited:
                visited.add(state)
                q.append(state)
    return False


def _default_bfs_fn(board: Board, r1: int, c1: int, r2: int, c2: int) -> list:
    """Wrapper để _bfs_check có cùng interface bfs_fn(board, r1,c1,r2,c2) -> list."""
    mat = board if isinstance(board, np.ndarray) else board_to_numpy(board)
    return [(r1, c1), (r2, c2)] if _bfs_check(mat, r1, c1, r2, c2) else []

def _call_bfs_fn(bfs_fn: Optional[BfsFn], board: Board, r1: int, c1: int, r2: int, c2: int):
    """Gọi bfs_fn theo 2 interface hỗ trợ: (board, r1, c1, r2, c2) và (r1, c1, r2, c2)."""
    fn = bfs_fn if bfs_fn is not None else _default_bfs_fn
    try:
        return fn(board, r1, c1, r2, c2)
    except TypeError:
        return fn(r1, c1, r2, c2)


def getRandomizedBoard(
    board_width: int = DEFAULT_BOARD_WIDTH,
    board_height: int = DEFAULT_BOARD_HEIGHT,
    num_same_heroes: int = 4,
) -> Board:
    """Tạo board mới với các cặp pokemon ngẫu nhiên, phù hợp cho môi trường Pikachu."""
    board = [[0 for _ in range(board_width)] for _ in range(board_height)]
    inner_cells = (board_height - 2) * (board_width - 2)
    if inner_cells <= 0:
        return board

    num_types = max(1, inner_cells // max(1, num_same_heroes))
    usable_cells = num_types * num_same_heroes
    pool = list(range(1, num_types + 1)) * num_same_heroes
    random.shuffle(pool)
    pool = pool[:usable_cells]

    index = 0
    for r in range(1, board_height - 1):
        for c in range(1, board_width - 1):
            if index < usable_cells:
                board[r][c] = pool[index]
                index += 1
    return board


def alterBoardWithLevel(board: Board, boxy1: int, boxx1: int, boxy2: int, boxx2: int, level: int) -> Board:
    """Dịch chuyển các pokemon theo level, đồng bộ với logic trong trò chơi gốc."""
    board = copy.deepcopy(board)
    if level <= 1:
        return board

    if level == 2:
        for col in {boxx1, boxx2}:
            values = [board[r][col] for r in range(len(board))]
            filtered = [value for value in values if value != 0]
            new_values = filtered + [0] * (len(board) - len(filtered))
            for r, value in enumerate(new_values):
                board[r][col] = value
        return board

    if level == 3:
        for col in {boxx1, boxx2}:
            values = [board[r][col] for r in range(len(board))]
            filtered = [value for value in values if value != 0]
            new_values = [0] * (len(board) - len(filtered)) + filtered
            for r, value in enumerate(new_values):
                board[r][col] = value
        return board

    if level == 4:
        for row in {boxy1, boxy2}:
            values = board[row]
            filtered = [value for value in values if value != 0]
            new_values = filtered + [0] * (len(values) - len(filtered))
            board[row] = new_values
        return board

    if level == 5:
        for row in {boxy1, boxy2}:
            values = board[row]
            filtered = [value for value in values if value != 0]
            new_values = [0] * (len(values) - len(filtered)) + filtered
            board[row] = new_values
        return board

    return board

# ══════════════════════════════════════════════════════════════════════
# PHẦN 3 — TÌM ACTION HỢP LỆ (ACTION MASKING)
# ══════════════════════════════════════════════════════════════════════

def get_valid_actions(board: Board, bfs_fn: Optional[BfsFn] = None) -> List[Action]:
    """
    Tìm tất cả cặp (r1,c1,r2,c2) có thể nối được trên bàn.

    Parameters
    ----------
    board  : board hiện tại
    bfs_fn : hàm bfs(board, r1, c1, r2, c2) -> list (path rỗng = không nối được).
             - Nếu có game thật (pikachu.py): truyền bfs thật vào để dùng
               đúng luật của game.
                   from pikachu import bfs
                   get_valid_actions(board, bfs)
             - Nếu KHÔNG truyền (None): tự dùng BFS nội bộ (_default_bfs_fn),
               không cần pygame, dùng tốt cho train offline / unit test.
    """
    fn = bfs_fn if bfs_fn is not None else _default_bfs_fn

    H = len(board)
    W = len(board[0]) if H > 0 else 0
    valid: List[Action] = []

    positions_by_type: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            val = board[r][c]
            if val != 0:
                positions_by_type.setdefault(val, []).append((r, c))

    for positions in positions_by_type.values():
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                r1, c1 = positions[i]
                r2, c2 = positions[j]
                if _call_bfs_fn(fn, board, r1, c1, r2, c2):
                    valid.append((r1, c1, r2, c2))

    return valid


def get_one_valid_action(board: Board, bfs_fn: Optional[BfsFn] = None) -> Optional[Action]:
    """Lấy cặp đầu tiên tìm được, dùng làm fallback cho agent. None nếu deadlock."""
    actions = get_valid_actions(board, bfs_fn)
    return actions[0] if actions else None


def has_valid_move(board: Union[Board, np.ndarray], bfs_fn: Optional[BfsFn] = None) -> bool:
    """Kiểm tra nhanh còn nước đi không (dừng sớm khi tìm thấy 1 cặp)."""
    b = board.tolist() if isinstance(board, np.ndarray) else board
    return get_one_valid_action(b, bfs_fn) is not None


# ══════════════════════════════════════════════════════════════════════
# PHẦN 4 — ENCODE / DECODE ACTION
# ══════════════════════════════════════════════════════════════════════

def action_to_pair_index(action: Action, valid_actions: List[Action]) -> int:
    """Vị trí của action trong danh sách valid_actions. -1 nếu không có."""
    try:
        return valid_actions.index(action)
    except ValueError:
        return -1


def pair_index_to_action(idx: int, valid_actions: List[Action]) -> Optional[Action]:
    """Nghịch đảo của action_to_pair_index. None nếu idx ngoài range."""
    if 0 <= idx < len(valid_actions):
        return valid_actions[idx]
    return None


def encode_action(action: Action, board_width: int, board_height: int) -> int:
    """
    Mã hoá action (r1,c1,r2,c2) -> số nguyên duy nhất (fixed encoding).
    encoded = (r1*W + c1) * (H*W) + (r2*W + c2)
    """
    r1, c1, r2, c2 = action
    W, H = board_width, board_height
    return (r1 * W + c1) * (H * W) + (r2 * W + c2)


def decode_action(encoded: int, board_width: int, board_height: int) -> Action:
    """Giải mã số nguyên -> (r1, c1, r2, c2). Nghịch đảo của encode_action."""
    W, H = board_width, board_height
    total = H * W
    idx1, idx2 = divmod(encoded, total)
    r1, c1 = divmod(idx1, W)
    r2, c2 = divmod(idx2, W)
    return (r1, c1, r2, c2)


def build_action_mask(valid_actions: List[Action], all_actions: List[Action]) -> np.ndarray:
    """
    Binary mask che action không hợp lệ trước khi argmax.
    Shape (len(all_actions),): 1.0 nếu hợp lệ, 0.0 nếu không.
    """
    valid_set = set(valid_actions)
    return np.array(
        [1.0 if a in valid_set else 0.0 for a in all_actions],
        dtype=np.float32,
    )


def qvalues_to_action(
    q_values: Union[np.ndarray, "torch.Tensor"],
    valid_actions: List[Action],
    board_width: int,
    board_height: int,
) -> Optional[Action]:
    """
    Chuyển Q-value tensor của model -> action hợp lệ tốt nhất.
    Chỉ xét Q-value của action trong valid_actions (masking cứng).

    q_values shape (N,)   : coi là điểm mỗi ô; score(i,j) = q[i] + q[j]
    q_values shape (N, N) : score(i,j) = q[i, j]
    """
    if not valid_actions:
        return None

    if _TORCH_AVAILABLE and isinstance(q_values, torch.Tensor):
        q_np = q_values.detach().cpu().numpy()
    else:
        q_np = np.array(q_values)

    best_action, best_score = None, -np.inf
    for action in valid_actions:
        r1, c1, r2, c2 = action
        i, j = r1 * board_width + c1, r2 * board_width + c2
        score = float(q_np[i]) + float(q_np[j]) if q_np.ndim == 1 else float(q_np[i, j])
        if score > best_score:
            best_score, best_action = score, action
    return best_action


def qvalues_to_action_topk(
    q_values: Union[np.ndarray, "torch.Tensor"],
    valid_actions: List[Action],
    board_width: int,
    k: int = 3,
) -> List[Tuple[Action, float]]:
    """Giống qvalues_to_action nhưng trả về top-k (action, score) giảm dần."""
    if not valid_actions:
        return []

    if _TORCH_AVAILABLE and isinstance(q_values, torch.Tensor):
        q_np = q_values.detach().cpu().numpy()
    else:
        q_np = np.array(q_values)

    scored = []
    for action in valid_actions:
        r1, c1, r2, c2 = action
        i, j = r1 * board_width + c1, r2 * board_width + c2
        score = float(q_np[i]) + float(q_np[j]) if q_np.ndim == 1 else float(q_np[i, j])
        scored.append((action, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ══════════════════════════════════════════════════════════════════════
# PHẦN 5 — TIỆN ÍCH ĐO LƯỜNG TRẠNG THÁI BOARD
# ══════════════════════════════════════════════════════════════════════

def count_remaining(board: Union[Board, np.ndarray]) -> int:
    """Đếm số ô còn pokemon (khác 0)."""
    if isinstance(board, np.ndarray):
        return int(np.count_nonzero(board))
    return sum(1 for row in board for cell in row if cell != 0)


def is_board_empty(board: Union[Board, np.ndarray]) -> bool:
    """True nếu board đã clear hoàn toàn."""
    return count_remaining(board) == 0


# Alias giữ tương thích tên cũ từ game_interface.py
is_game_complete = is_board_empty


def board_progress(board: Board, initial_count: int) -> float:
    """
    % tiến độ xoá bàn, trong [0.0, 1.0].
        reward += board_progress(board, initial_count) * PROGRESS_BONUS
    """
    if initial_count == 0:
        return 1.0
    return 1.0 - count_remaining(board) / initial_count


def get_board_info(board: Union[Board, np.ndarray], bfs_fn: Optional[BfsFn] = None) -> Dict:
    """Tóm tắt trạng thái bảng, dùng để log/debug."""
    b = board.tolist() if isinstance(board, np.ndarray) else board
    info = get_state_info(b)
    actions = get_valid_actions(b, bfs_fn)
    return {
        "remaining_cells": info["num_remaining"],
        "unique_types"   : info["num_types"],
        "valid_actions"  : len(actions),
        "is_complete"    : info["num_remaining"] == 0,
        "has_valid_move" : len(actions) > 0,
    }


# ══════════════════════════════════════════════════════════════════════
# PHẦN 6 — APPLY ACTION / REWARD SHAPING (mô phỏng offline, train)
# ══════════════════════════════════════════════════════════════════════

def apply_action(board: Board, action: Action, bfs_fn: Optional[BfsFn] = None) -> Optional[Board]:
    """
    Áp dụng action (r1,c1,r2,c2) lên board (deepcopy, không sửa board gốc).
    Trả về board mới, hoặc None nếu action không hợp lệ.
    """
    r1, c1, r2, c2 = action
    new_board = copy.deepcopy(board)
    fn = bfs_fn if bfs_fn is not None else _default_bfs_fn

    if not _call_bfs_fn(fn, new_board, r1, c1, r2, c2):
        return None

    new_board[r1][c1] = 0
    new_board[r2][c2] = 0
    return new_board


def compute_reward(
    board_before: Board,
    board_after: Optional[Board],
    action_valid: bool,
    game_done: bool,
    remaining_time: float = 0.0,
) -> float:
    """
    Reward cho một bước đi của Agent:
        +10   xoá được cặp hợp lệ
        -5    action không hợp lệ
        +100  hoàn thành bảng (+0.01 * thời gian còn lại)
        -1    step penalty (khuyến khích đi nhanh)
    """
    reward = -1.0
    if not action_valid:
        return reward - 5.0
    reward += 10.0
    if game_done:
        reward += 100.0 + 0.01 * remaining_time
    return reward


# ══════════════════════════════════════════════════════════════════════
# PHẦN 7 — EXTRACT STATE ĐẦY ĐỦ (board + meta) CHO TRAINING
# ══════════════════════════════════════════════════════════════════════

def extract_state(
    board: Board,
    remaining_time: float = MAX_TIME,
    lives: int = MAX_LIVES,
    score: int = 0,
    level: int = 1,
    max_id: int = MAX_POKEMON_ID,
) -> Dict[str, np.ndarray]:
    """
    Trích xuất toàn bộ trạng thái ván chơi thành dict numpy array:
        board_raw    (H, W)     int32
        board_onehot (C, H, W)  float32  -> CNN
        board_norm   (1, H, W)  float32  -> CNN 1 channel
        meta         (4,)       float32  [time, lives, score, level] chuẩn hoá
    """
    raw    = board_to_numpy(board)
    onehot = board_to_onehot(board, max_id + 1)
    norm   = normalize_board(board, max_id)[np.newaxis, :]

    meta = np.array(
        [
            remaining_time / MAX_TIME,
            lives / MAX_LIVES,
            score / MAX_SCORE,
            level / MAX_LEVEL,
        ],
        dtype=np.float32,
    )

    return {"board_raw": raw, "board_onehot": onehot, "board_norm": norm, "meta": meta}


def state_to_flat_vector(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten state dict thành vector 1D (H*W + 4,) cho MLP đơn giản."""
    board_flat = state["board_norm"].flatten()
    return np.concatenate([board_flat, state["meta"]])


def state_to_torch(state: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, "torch.Tensor"]:
    """Chuyển dict state (numpy) -> dict tensor PyTorch, input trực tiếp cho model."""
    assert _TORCH_AVAILABLE, "PyTorch chưa được cài đặt. Chạy: pip install torch"
    dev = torch.device(device)
    board_flat = np.concatenate([state["board_norm"].flatten(), state["meta"]]).astype(np.float32)

    return {
        "board_raw"   : torch.from_numpy(state["board_raw"]).long().to(dev),
        "board_onehot": torch.from_numpy(state["board_onehot"]).float().to(dev),
        "board_norm"  : torch.from_numpy(state["board_norm"]).float().to(dev),
        "meta"        : torch.from_numpy(state["meta"]).float().to(dev),
        "board_flat"  : torch.from_numpy(board_flat).float().to(dev),
    }


def batch_states_to_torch(states: List[Dict[str, np.ndarray]], device: str = "cpu") -> Dict[str, "torch.Tensor"]:
    """Gộp danh sách state thành batch tensor (dùng cho Replay Memory / mini-batch)."""
    assert _TORCH_AVAILABLE, "PyTorch chưa được cài đặt."
    dev = torch.device(device)

    def _stack(key):
        return np.stack([s[key] for s in states], axis=0)

    onehot = _stack("board_onehot")
    norm   = _stack("board_norm")
    meta   = _stack("meta")
    flat   = np.concatenate([norm.reshape(len(states), -1), meta], axis=1)

    return {
        "board_onehot": torch.from_numpy(onehot).float().to(dev),
        "board_norm"  : torch.from_numpy(norm).float().to(dev),
        "meta"        : torch.from_numpy(meta).float().to(dev),
        "board_flat"  : torch.from_numpy(flat).float().to(dev),
    }


# ══════════════════════════════════════════════════════════════════════
# PHẦN 8 — DEMO / QUICK TEST (chạy trực tiếp: python utils/state_utils.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_board = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 1, 3, 0],
        [0, 2, 3, 4, 4, 0],
        [0, 1, 3, 2, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    H, W = 5, 6

    print("=" * 62)
    print("DEMO state_utils.py (bản gộp)")
    print("=" * 62)

    raw = board_to_numpy(demo_board)
    print("\n[1] board_to_numpy:\n", raw)

    onehot = board_to_onehot(demo_board, num_classes=5)
    print("\n[2] board_to_onehot shape:", onehot.shape)

    info = get_state_info(demo_board)
    print("\n[3] get_state_info:", info)

    # KHÔNG truyền bfs_fn -> tự dùng BFS nội bộ (không cần game thật)
    actions = get_valid_actions(demo_board)
    print(f"\n[4] get_valid_actions -> {len(actions)} cặp:")
    for a in actions:
        print(f"    {a}  id={demo_board[a[0]][a[1]]}")

    if actions:
        enc = encode_action(actions[0], board_width=W, board_height=H)
        dec = decode_action(enc, board_width=W, board_height=H)
        print(f"\n[5] encode_action({actions[0]}) -> {enc} -> decode -> {dec}")

        new_board = apply_action(demo_board, actions[0])
        print(f"\n[6] apply_action xoá {actions[0]}:\n", np.array(new_board))

        reward = compute_reward(demo_board, new_board, action_valid=True,
                                 game_done=is_board_empty(new_board))
        print(f"\n[7] compute_reward:", reward)

    state = extract_state(demo_board, remaining_time=200.0, lives=8, score=15, level=2, max_id=4)
    print("\n[8] extract_state keys:", list(state.keys()), " meta:", state["meta"])

    if _TORCH_AVAILABLE:
        tensors = state_to_torch(state)
        print("\n[9] state_to_torch shapes:")
        for k, v in tensors.items():
            print(f"    {k:14s}: {tuple(v.shape)}  {v.dtype}")
    else:
        print("\n[9] PyTorch chưa cài -> bỏ qua test tensor.")

    print("\n[10] get_board_info:", get_board_info(demo_board))