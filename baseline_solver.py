import numpy as np
from path_validator import find_valid_path

def baseline_choose_move(board):
    """
    Quét toàn bộ bàn cờ để tìm một cặp ô hợp lệ có thể nối được.
    Trả về tuple ((r1, c1), (r2, c2)) nếu tìm thấy, ngược lại trả về None.
    """
    rows, cols = board.shape
    
    # Quét từng ô trên bàn cờ làm điểm bắt đầu (ô A)
    for r1 in range(rows):
        for c1 in range(cols):
            val1 = board[r1, c1]
            
            # Bỏ qua nếu là ô trống
            if val1 == -1:
                continue
                
            # Quét các ô còn lại làm điểm kết thúc (ô B)
            for r2 in range(rows):
                for c2 in range(cols):
                    # Điều kiện (r1, c1) >= (r2, c2) giúp tránh so sánh trùng một ô 
                    # và không bị lặp lại các cặp đã kiểm tra (ví dụ A nối B thì không cần check B nối A)
                    if (r1, c1) >= (r2, c2):
                        continue
                        
                    val2 = board[r2, c2]
                    
                    # Chỉ kiểm tra đường đi nếu 2 ô có cùng ID Pokémon
                    if val1 == val2:
                        path = find_valid_path(board, (r1, c1), (r2, c2))
                        if path is not None:
                            # Trả về ngay cặp đầu tiên hợp lệ tìm được
                            return ((r1, c1), (r2, c2))
                            
    # Nếu chạy hết 4 vòng lặp mà không return được gì -> Bàn cờ bế tắc
    return None

def apply_move(board, cellA, cellB):
    """
    'Xoá' một cặp ô đã nối thành công khỏi bàn cờ (gán về -1, tức ô trống).
    Trả về board mới (không sửa board gốc, để dễ debug/undo nếu cần).
    """
    new_board = board.copy()
    new_board[cellA] = -1
    new_board[cellB] = -1
    return new_board


def is_board_cleared(board):
    """Bàn cờ đã sạch hoàn toàn (thắng) khi không còn ô nào khác -1."""
    return bool(np.all(board == -1))


def play_full_game(board, verbose=True, max_steps=10000):
    """
    Chạy baseline_choose_move LIÊN TỤC cho đến khi:
      - Bàn cờ sạch hoàn toàn (thắng), hoặc
      - Không còn cặp nào nối được nữa (bế tắc).

    Đây là tiêu chí hoàn thành của TV4: "baseline_solver.py chạy được độc lập,
    chọn được nước đi hợp lệ liên tục đến khi bàn cờ hết nước đi."

    Trả về dict thống kê: {
        "cleared": bool,        # True nếu xoá sạch bàn cờ
        "moves": list,          # danh sách các nước đi đã thực hiện theo thứ tự
        "remaining": int,       # số ô còn lại lúc dừng (0 nếu thắng)
        "steps": int,           # tổng số nước đã đi
        "final_board": ndarray, # trạng thái bàn cờ lúc dừng
    }
    """
    current = board.copy()
    moves = []
    step = 0

    while step < max_steps:
        if is_board_cleared(current):
            break

        move = baseline_choose_move(current)
        if move is None:
            # Bế tắc: không còn cặp nào nối được nữa
            break

        cellA, cellB = move
        current = apply_move(current, cellA, cellB)
        moves.append(move)
        step += 1

        if verbose:
            remaining = int(np.sum(current != -1))
            print(f"  Bước {step:3d}: nối {cellA} <-> {cellB}  "
                  f"(còn lại {remaining} ô)")

    cleared = is_board_cleared(current)
    remaining = int(np.sum(current != -1))

    result = {
        "cleared": cleared,
        "moves": moves,
        "remaining": remaining,
        "steps": step,
        "final_board": current,
    }

    if verbose:
        print()
        if cleared:
            print(f"=> THẮNG! Đã xoá sạch bàn cờ sau {step} nước đi.")
        else:
            print(f"=> BẾ TẮC sau {step} nước đi. Còn lại {remaining} ô chưa xoá được.")
            print("   Bàn cờ lúc dừng:")
            print(current)

    return result


def generate_solvable_board(rows, cols, n_types, seed=None):
    """
    Sinh một bàn cờ NGẪU NHIÊN nhưng LUÔN giải được 100%: tạo từng cặp ID
    giống nhau và đặt ngẫu nhiên vào các ô trống còn lại, đảm bảo rows*cols
    là số chẵn. Hữu ích để test play_full_game với board luôn "cleared".
    """
    rng = np.random.default_rng(seed)
    total_cells = rows * cols
    if total_cells % 2 != 0:
        raise ValueError("rows*cols phải là số chẵn để mọi ô đều có cặp")

    n_pairs = total_cells // 2
    # Phân bổ n_pairs cặp đều cho n_types loại (tuần hoàn nếu không chia hết)
    ids = [(i % n_types) + 1 for i in range(n_pairs)]
    values = np.repeat(ids, 2)  # mỗi ID xuất hiện đúng 2 lần
    rng.shuffle(values)
    return values.reshape(rows, cols)


if __name__ == "__main__":
    # Test 1: Dùng lại bàn cờ test chuẩn (4x4) - chạy đến khi bế tắc hoặc thắng
    board = np.array([
        [ 1, -1,  2,  1],
        [-1,  3, -1, -1],
        [ 2,  3,  4, -1],
        [ 5, -1, -1,  4]
    ])

    print("=== VÁN 1: BÀN CỜ TEST CHUẨN (4x4) ===")
    print(board)
    print()
    result1 = play_full_game(board)
    print(f"\nTổng kết: cleared={result1['cleared']}, "
          f"steps={result1['steps']}, remaining={result1['remaining']}\n")

    print("=" * 50)
    print()

    # Test 2: Bàn cờ luôn giải được 100% (sinh ngẫu nhiên có đảm bảo lời giải)
    print("=== VÁN 2: BÀN CỜ NGẪU NHIÊN LUÔN GIẢI ĐƯỢC (6x4) ===")
    solvable_board = generate_solvable_board(rows=6, cols=4, n_types=6, seed=7)
    print(solvable_board)
    print()
    result2 = play_full_game(solvable_board)
    print(f"\nTổng kết: cleared={result2['cleared']}, "
          f"steps={result2['steps']}, remaining={result2['remaining']}")

    if not result2["cleared"]:
        print("LƯU Ý: Baseline 'tham lam chọn cặp đầu tiên' không đảm bảo giải "
              "được mọi board có lời giải lý thuyết — đây là hành vi đã biết "
              "(baseline không nhìn trước/backtrack), không phải lỗi.")