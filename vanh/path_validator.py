from collections import deque
import numpy as np

# Định nghĩa 4 hướng di chuyển: Lên, Xuống, Trái, Phải
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_walkable(board, r, c):
    """
    Kiểm tra xem một toạ độ (r, c) có thể đi qua được không.
    Cho phép đi ra ngoài viền bàn cờ 1 ô theo mọi trục, KỂ CẢ góc ảo kép
    (vd r=-1 và c=-1 cùng lúc) — đây là luật Onet chuẩn, dùng để "cắt góc"
    khi nối 2 ô ở góc chéo bàn cờ chỉ với 2 lần rẽ.
    """
    rows, cols = board.shape

    # Nếu cả 2 trục đều nằm trong dải mở rộng [-1, rows]/[-1, cols],
    # và ít nhất 1 trục nằm ngoài bàn cờ thật -> luôn đi được (viền ảo, kể cả góc)
    r_ok = -1 <= r <= rows
    c_ok = -1 <= c <= cols
    if not (r_ok and c_ok):
        return False  # văng ra quá xa viền ảo

    r_in_real = 0 <= r < rows
    c_in_real = 0 <= c < cols
    if r_in_real and c_in_real:
        # Nằm hẳn trong bàn cờ thật -> phải là ô trống
        return board[r, c] == -1

    # Còn lại: nằm trên viền ảo (kể cả góc ảo kép) -> luôn đi được
    return True

def find_valid_path(board, start, end, max_turns=2):
    """
    Tìm đường đi từ start đến end với tối đa max_turns lần rẽ.
    Trả về list toạ độ đường đi nếu hợp lệ, ngược lại trả về None.
    """
    # Queue lưu trạng thái: (hàng, cột, hướng_đang_đi, số_lần_đã_rẽ, lịch_sử_đường_đi)
    queue = deque()
    # Tập hợp các trạng thái đã duyệt để tránh lặp vô hạn.
    # QUAN TRỌNG: phải đánh dấu visited NGAY KHI PUSH vào queue, không phải lúc pop.
    # Nếu đánh dấu trễ (lúc pop), nhiều bản sao của cùng 1 trạng thái (r, c, d_idx, turns)
    # có thể bị đẩy vào queue nhiều lần trước khi bản đầu tiên kịp bị lọc, khiến BFS
    # nổ số lượng theo cấp số nhân và treo vô hạn trên các board không có lời giải
    # (vd: bò vòng vô tận dọc viền ảo quanh bàn cờ).
    visited = set()
    
    # Khởi tạo bước đi đầu tiên từ điểm xuất phát ra 4 hướng
    sr, sc = start
    for d_idx, (dr, dc) in enumerate(DIRECTIONS):
        nr, nc = sr + dr, sc + dc
        # Nếu bước đầu tiên đã chạm đích, trả về ngay
        if (nr, nc) == end:
            return [start, (nr, nc)]
        # Nếu ô tiếp theo đi được, ném vào queue (và đánh dấu visited ngay)
        if is_walkable(board, nr, nc):
            state_key = (nr, nc, d_idx, 0)
            if state_key not in visited:
                visited.add(state_key)
                queue.append((nr, nc, d_idx, 0, [start, (nr, nc)]))

    # Bắt đầu vòng lặp BFS
    while queue:
        r, c, d_idx, turns, path = queue.popleft()

        # Từ vị trí hiện tại, tiếp tục thử rẽ sang 4 hướng
        for nd_idx, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            
            # Tính số lần rẽ: Nếu hướng mới (nd_idx) khác hướng cũ (d_idx) -> cộng thêm 1 lần rẽ
            new_turns = turns + (1 if nd_idx != d_idx else 0)
            
            # Nếu vượt quá giới hạn số lần rẽ (mặc định > 2) thì huỷ nhánh này
            if new_turns > max_turns:
                continue
                
            # Nếu bước tiếp theo là đích, nối thêm vào path và trả về ngay
            if (nr, nc) == end:
                return path + [(nr, nc)]
                
            # Nếu bước tiếp theo đi được, kiểm tra & đánh dấu visited NGAY LÚC PUSH
            if is_walkable(board, nr, nc):
                state_key = (nr, nc, nd_idx, new_turns)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((nr, nc, nd_idx, new_turns, path + [(nr, nc)]))
                
    # Nếu duyệt hết queue mà vẫn không chốt được đích -> Bế tắc
    return None
def _validate_path_shape(board, path, max_turns=2):
    """
    Helper kiểm tra một path có thực sự hợp lệ không (dùng cho test):
    - Các bước liên tiếp phải kề nhau (Manhattan distance == 1)
    - Số lần rẽ không vượt quá max_turns
    - Mọi ô ở giữa (không tính điểm đầu/cuối) phải đi được (is_walkable)
    """
    if path is None or len(path) < 2:
        return False, "Path rỗng hoặc quá ngắn"

    turns = 0
    prev_dir = None
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        dr, dc = r1 - r0, c1 - c0
        if abs(dr) + abs(dc) != 1:
            return False, f"Bước {i} không liền kề: {path[i-1]} -> {path[i]}"
        cur_dir = (dr, dc)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
        # Ô giữa đường (không phải điểm cuối) phải walkable
        if i != len(path) - 1 and not is_walkable(board, r1, c1):
            return False, f"Ô giữa đường không đi được: {path[i]}"

    if turns > max_turns:
        return False, f"Vượt quá số lần rẽ cho phép: {turns} > {max_turns}"

    return True, f"OK ({turns} lần rẽ)"


def run_test_suite():
    """
    Bộ 8 test case biên theo yêu cầu Bước 3 (TV4):
    - 2 ô liền kề cùng hàng/cột
    - 1 lần rẽ (chữ L)
    - 2 lần rẽ qua viền ngoài
    - 2 ô bị chặn hoàn toàn (None)
    - Bàn cờ gần kín chỉ còn 1 đường hẹp
    - Thêm: 2 lần rẽ không qua viền, giới hạn max_turns chặt
    """
    board = np.array([
        [ 1, -1,  2,  1], # Hàng 0
        [-1,  3, -1, -1], # Hàng 1
        [ 2,  3,  4, -1], # Hàng 2
        [ 5, -1, -1,  4]  # Hàng 3
    ])

    print("=== BÀN CỜ TEST CHUẨN (4x4) ===")
    print(board)
    print("================================\n")

    tests = []

    # Test 1: Hai ô liền kề cùng cột (0 lần rẽ) - ID 3 ở (1,1) và (2,1)
    tests.append(("Liền kề cùng cột (0 rẽ)", board, (1, 1), (2, 1), True))

    # Test 2: Hai ô liền kề cùng hàng (0 lần rẽ)
    board_row_adj = np.array([
        [ 7,  7, -1],
        [-1, -1, -1],
        [-1, -1, -1]
    ])
    tests.append(("Liền kề cùng hàng (0 rẽ)", board_row_adj, (0, 0), (0, 1), True))

    # Test 3: Đi theo hình chữ L (1 lần rẽ) - ID 4 ở (2,2) và (3,3)
    tests.append(("Chữ L (1 rẽ)", board, (2, 2), (3, 3), True))

    # Test 4: Đi vòng qua viền ngoài (2 lần rẽ) - ID 1 ở (0,0) và (0,3)
    tests.append(("Qua viền ngoài (2 rẽ)", board, (0, 0), (0, 3), True))

    # Test 5: Hai ô bị chặn hoàn toàn (Bế tắc, trả None)
    tests.append(("Bị chặn hoàn toàn (None)", board, (0, 2), (2, 0), False))

    # Test 6: 2 lần rẽ KHÔNG qua viền, đi thẳng trong bàn cờ
    board_2turn_inside = np.array([
        [ 9, -1, -1],
        [-1, -1, -1],
        [-1, -1,  9]
    ])
    tests.append(("2 rẽ trong bàn cờ (không qua viền)", board_2turn_inside, (0, 0), (2, 2), True))

    # Test 7: Bàn cờ gần kín, chỉ còn đúng 1 đường hẹp duy nhất (hành lang cột giữa)
    board_narrow = np.array([
        [ 6,  8,  6],
        [ 6, -1,  6],
        [ 6, -1,  6],
        [ 6,  8,  6]
    ])
    tests.append(("Đường hẹp duy nhất qua hành lang", board_narrow, (0, 1), (3, 1), True))

    # Test 8: Giới hạn max_turns=0, kỳ vọng None vì đường duy nhất cần rẽ 1 lần (chữ L)
    board_zigzag = np.array([
        [ 5, -1,  6],
        [ 6, -1, -1],
        [ 6,  6,  5],
    ])
    tests.append(("Giới hạn max_turns=0 (chặt)", board_zigzag, (0, 0), (2, 2), False))

    passed, failed = 0, 0
    for i, t in enumerate(tests, 1):
        name, b, start, end, expect_found = t
        max_turns = 0 if "max_turns=0" in name else 2
        path = find_valid_path(b, start, end, max_turns=max_turns)
        found = path is not None

        status = "FAIL"
        detail = ""
        if found != expect_found:
            detail = f"(kỳ vọng {'có đường' if expect_found else 'None'}, nhận {'có đường' if found else 'None'})"
        elif found:
            ok, msg = _validate_path_shape(b, path, max_turns=max_turns)
            if ok:
                status = "PASS"
                detail = msg
            else:
                detail = f"Path trả về SAI LUẬT: {msg}"
        else:
            status = "PASS"
            detail = "Đúng là bế tắc (None)"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"Test {i} [{name}]: {start} -> {end}")
        print(f"  [{status}] {detail}")
        if path is not None:
            print(f"  Path: {path}")
        print()

    print(f"=== KẾT QUẢ: {passed}/{len(tests)} PASS, {failed} FAIL ===\n")
    return passed, failed


def run_benchmark(board_shape=(8, 6), n_calls=300, seed=42):
    """
    Đo tốc độ find_valid_path trên board kích thước thật (mặc định 8x6 theo config.py của TV2).
    Tiêu chí hoàn thành: < 50ms / lần gọi.
    """
    import time
    import random

    rng = random.Random(seed)
    rows, cols = board_shape

    board = -np.ones((rows, cols), dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    n_filled = int(len(cells) * 0.4)
    for idx in range(n_filled):
        r, c = cells[idx]
        board[r, c] = rng.randint(1, 6)

    print(f"=== BENCHMARK TỐC ĐỘ ({rows}x{cols}, {n_calls} lần gọi) ===")
    print(board)
    print()

    filled_cells = [(r, c) for r in range(rows) for c in range(cols) if board[r, c] != -1]
    pairs = []
    for _ in range(n_calls):
        if len(filled_cells) >= 2:
            a, b = rng.sample(filled_cells, 2)
        else:
            a, b = filled_cells[0], filled_cells[0]
        pairs.append((a, b))

    durations_ms = []
    for a, b in pairs:
        t0 = time.perf_counter()
        find_valid_path(board, a, b)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000)

    durations_ms.sort()
    avg = sum(durations_ms) / len(durations_ms)
    p50 = durations_ms[len(durations_ms) // 2]
    p95 = durations_ms[int(len(durations_ms) * 0.95)]
    worst = durations_ms[-1]

    print(f"Trung bình : {avg:.3f} ms")
    print(f"P50        : {p50:.3f} ms")
    print(f"P95        : {p95:.3f} ms")
    print(f"Tệ nhất    : {worst:.3f} ms")

    target_ms = 50.0
    if worst < target_ms:
        print(f"=> ĐẠT tiêu chí (<{target_ms}ms mỗi lần gọi)\n")
    else:
        print(f"=> CHƯA ĐẠT tiêu chí (<{target_ms}ms), cân nhắc nâng cấp A*\n")

    return avg, worst


if __name__ == "__main__":
    p, f = run_test_suite()
    run_benchmark(board_shape=(8, 6), n_calls=300)

    if f > 0:
        raise SystemExit(1)