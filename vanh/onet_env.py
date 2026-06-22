"""
onet_env.py
=============
Đây là MÔI TRƯỜNG GIẢ LẬP (simulated environment) của game Onet.

TẠI SAO CẦN FILE NÀY?
- TV2 sẽ cho bạn icon ảnh thật từ app, nhưng AI không "học" trực tiếp trên app thật được
  (vì train cần chơi hàng nghìn ván, mà app thật load chậm + không reset nhanh được).
- Nên ta GIẢ LẬP bàn cờ bằng số nguyên trong NumPy: mỗi ô là 1 con số (loại Pokémon),
  -1 nghĩa là ô trống. AI học trên dữ liệu giả lập này trước, sau đó dùng model đã học
  để "suy luận" (predict) khi chạy trên app thật.

FORMAT DỮ LIỆU (đây là "hợp đồng" mà TV1, TV4 cũng phải dùng chung):
- board: np.ndarray shape (rows, cols), giá trị nguyên từ 0..N-1 là loại Pokémon, -1 là ô trống.
- action: tuple ((r1, c1), (r2, c2)) - toạ độ 2 ô muốn nối.
"""

import numpy as np
import random


class OnetEnv:
    def __init__(self, rows=6, cols=6, n_types=6, validator=None):
        """
        rows, cols: kích thước bàn cờ. Bắt đầu để NHỎ (vd 4x4 hoặc 6x6) cho dễ học,
                    sau khi chắc code chạy đúng mới tăng dần lên kích thước thật.
        n_types: số loại Pokémon khác nhau trên bàn cờ.
        validator: hàm kiểm tra đường nối hợp lệ, có chữ ký
                   validator(board, cellA, cellB) -> True/False
                   Đây là module của TV4. Nếu chưa có, dùng bản giả lập đơn giản bên dưới.
        """
        assert (rows * cols) % 2 == 0, "Tổng số ô phải là số chẵn (mỗi loại luôn có cặp)"
        self.rows = rows
        self.cols = cols
        self.n_types = n_types
        self.validator = validator if validator is not None else self._default_validator
        self.board = None
        self.reset()

    # ----------------------------------------------------------------
    # SINH BÀN CỜ MỚI (luôn đảm bảo có lời giải vì sinh theo CẶP)
    # ----------------------------------------------------------------
    def reset(self):
        total_cells = self.rows * self.cols
        n_pairs = total_cells // 2

        # Sinh ngẫu nhiên loại cho từng cặp, đảm bảo mỗi loại xuất hiện số chẵn lần
        types = [random.randint(0, self.n_types - 1) for _ in range(n_pairs)]
        cell_values = types + types  # nhân đôi để mỗi loại có đúng 1 cặp (ít nhất)
        random.shuffle(cell_values)

        self.board = np.array(cell_values, dtype=int).reshape(self.rows, self.cols)
        return self.get_state()

    # ----------------------------------------------------------------
    # VALIDATOR GIẢ LẬP TẠM THỜI (dùng khi TV4 chưa xong)
    # Phiên bản này kiểm tra: đường thẳng (0 lần rẽ) HOẶC đường chữ L (1 lần rẽ).
    # Đây vẫn là luật ĐƠN GIẢN HÓA so với luật Onet chuẩn (có thể có 2 lần rẽ qua viền
    # ngoài bàn cờ), nhưng đã đủ gần thực tế để demo training có ý nghĩa.
    # Khi TV4 xong, thay self.validator bằng hàm find_valid_path() thật của TV4
    # (hỗ trợ đầy đủ tối đa 2 lần rẽ + đi vòng qua viền ngoài bàn cờ).
    # ----------------------------------------------------------------
    def _default_validator(self, board, cellA, cellB):
        r1, c1 = cellA
        r2, c2 = cellB
        if board[r1, c1] == -1 or board[r2, c2] == -1:
            return False  # một trong hai ô đã trống rồi, không nối được
        if board[r1, c1] != board[r2, c2]:
            return False  # khác loại Pokémon, không nối được

        # Trường hợp 1: đường thẳng (0 lần rẽ) - cùng hàng hoặc cùng cột
        if r1 == r2:
            lo, hi = sorted([c1, c2])
            if np.all(board[r1, lo + 1:hi] == -1):
                return True
        elif c1 == c2:
            lo, hi = sorted([r1, r2])
            if np.all(board[lo + 1:hi, c1] == -1):
                return True

        # Trường hợp 2: đường chữ L (1 lần rẽ) qua 1 trong 2 điểm góc trung gian
        # Góc 1: (r1, c2) -- đi từ A ngang tới cột của B, rồi đi dọc xuống B
        corner1 = (r1, c2)
        if board[corner1] == -1 or corner1 == cellB:
            path_ok = True
            lo, hi = sorted([c1, c2])
            if not np.all(board[r1, lo + 1:hi] == -1) if c1 != c2 else False:
                path_ok = False
            lo2, hi2 = sorted([r1, r2])
            if path_ok and not (np.all(board[lo2 + 1:hi2, c2] == -1) if r1 != r2 else True):
                path_ok = False
            if path_ok:
                return True

        # Góc 2: (r2, c1) -- đi từ A dọc xuống hàng của B, rồi đi ngang tới B
        corner2 = (r2, c1)
        if board[corner2] == -1 or corner2 == cellB:
            path_ok = True
            lo, hi = sorted([r1, r2])
            if not np.all(board[lo + 1:hi, c1] == -1) if r1 != r2 else False:
                path_ok = False
            lo2, hi2 = sorted([c1, c2])
            if path_ok and not (np.all(board[r2, lo2 + 1:hi2] == -1) if c1 != c2 else True):
                path_ok = False
            if path_ok:
                return True

        return False

    # ----------------------------------------------------------------
    # LẤY TRẠNG THÁI (STATE) ĐỂ ĐƯA VÀO AI
    # ----------------------------------------------------------------
    def get_state(self):
        """Trả về bản sao bàn cờ hiện tại."""
        return self.board.copy()

    # ----------------------------------------------------------------
    # LIỆT KÊ CÁC ACTION HỢP LỆ (ACTION MASKING)
    # Đây là kỹ thuật quan trọng: thay vì xét MỌI cặp ô (hàng nghìn tổ hợp),
    # ta chỉ xét các cặp CÙNG LOẠI và còn tồn tại trên bàn cờ.
    # ----------------------------------------------------------------
    def get_valid_actions(self):
        valid = []
        # nhóm vị trí các ô theo loại Pokémon để khỏi so hết O(n^2) toàn bàn
        positions_by_type = {}
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.board[r, c]
                if val == -1:
                    continue
                positions_by_type.setdefault(val, []).append((r, c))

        for type_id, positions in positions_by_type.items():
            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    cellA, cellB = positions[i], positions[j]
                    if self.validator(self.board, cellA, cellB):
                        valid.append((cellA, cellB))
        return valid

    # ----------------------------------------------------------------
    # THỰC HIỆN MỘT BƯỚC (STEP) - đây là hàm quan trọng nhất của Environment
    # Input: action = ((r1,c1), (r2,c2))
    # Output: (state_mới, reward, done, info)
    # ----------------------------------------------------------------
    def step(self, action):
        cellA, cellB = action
        r1, c1 = cellA
        r2, c2 = cellB

        is_valid = self.validator(self.board, cellA, cellB)

        if not is_valid:
            # Phạt khi chọn action sai luật
            reward = -5
            done = False
            return self.get_state(), reward, done, {"valid": False}

        # Nối hợp lệ -> xoá 2 ô khỏi bàn cờ
        # Thưởng thêm nếu đây là cặp "khó" (ít lựa chọn nối khác đi kèm) để khuyến khích
        # AI ưu tiên xử lý ô khó trước, tránh để dồn lại gây bế tắc về sau.
        valid_actions_before = self.get_valid_actions()
        difficulty_bonus = max(0, 5 - len(valid_actions_before))  # càng ít lựa chọn càng thưởng nhiều

        self.board[r1, c1] = -1
        self.board[r2, c2] = -1

        remaining = np.sum(self.board != -1)

        if remaining == 0:
            # Xoá hết bàn cờ -> thắng lớn
            reward = 100
            done = True
            return self.get_state(), reward, done, {"valid": True, "win": True}

        # Kiểm tra bế tắc: hết nước đi nhưng bàn chưa trống
        next_valid_actions = self.get_valid_actions()
        if len(next_valid_actions) == 0:
            reward = -20  # phạt vì dẫn tới bế tắc
            done = True
            return self.get_state(), reward, done, {"valid": True, "win": False, "stuck": True}

        reward = 10 + difficulty_bonus
        done = False
        return self.get_state(), reward, done, {"valid": True}

    def render(self):
        """In bàn cờ ra màn hình dạng text, để xem trực quan khi debug."""
        for row in self.board:
            print(" ".join(f"{v:2d}" if v != -1 else " ." for v in row))
        print()


# ----------------------------------------------------------------
# TEST NHANH: chạy thử file này độc lập để chắc Environment hoạt động đúng
# Chạy bằng lệnh: python onet_env.py
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Sau khi có file của TV4 — copy file path_validator.py vào cùng thư mục tv3_final
    from path_validator import find_valid_path
    env = OnetEnv(rows=8, cols=6, n_types=8, validator=find_valid_path)
    print("Bàn cờ ban đầu:")
    env.render()

    valid_actions = env.get_valid_actions()
    print(f"Số action hợp lệ ban đầu: {len(valid_actions)}")

    # Thử nối 1 cặp ngẫu nhiên hợp lệ
    if valid_actions:
        action = random.choice(valid_actions)
        print(f"Thử nối: {action}")
        state, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        env.render()
