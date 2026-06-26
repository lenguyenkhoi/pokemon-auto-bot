import pygame
import numpy as np
import random

class PikachuEnv:
    def __init__(self, rows=8, cols=12):
        self.rows = rows
        self.cols = cols
        self.score = 0
        
        # Giả định game có 20 loại Pokemon khác nhau (ID từ 1 đến 20, 0 là ô trống)
        self.num_pokemon_types = 20 
        
        # Khởi tạo ma trận bàn cờ rỗng (bao gồm cả viền bao quanh để chạy thuật toán nối đường)
        self.state = np.zeros((self.rows + 2, self.cols + 2), dtype=np.int32)
        
        # Khởi tạo Pygame ngầm (Không cần vòng lặp sự kiện của con người)
        pygame.init()
        self.reset()

    def reset(self):
        """
        Nhiệm vụ 2: Tạo map mới ngẫu nhiên mỗi vòng chơi.
        Đảm bảo các cặp Pokemon luôn xuất hiện theo cặp chẵn để có thể phá đảo.
        """
        self.score = 0
        total_cells = self.rows * self.cols
        
        # 1. Tạo danh sách các cặp Pokemon chẵn
        half_cells = total_cells // 2
        pokemon_pool = []
        for _ in range(half_cells):
            pokemon_id = random.randint(1, self.num_pokemon_types)
            pokemon_pool.extend([pokemon_id, pokemon_id]) # Thêm 1 cặp giống nhau
            
        # 2. Trộn ngẫu nhiên danh sách
        random.shuffle(pokemon_pool)
        
        # 3. Đưa vào vùng lõi của ma trận state (bỏ phần viền 0 ngoài cùng)
        pokemon_idx = 0
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                self.state[r][c] = pokemon_pool[pokemon_idx]
                pokemon_idx += 1
                
        # Trả về trạng thái ban đầu của bàn cờ
        return self.state

    def _check_path_internal(self, p1, p2):
        """
        Gọi thuật toán check đường dẫn nội tại của game (do Người 1 phân tích từ mã nguồn gốc).
        p1, p2 là bộ tọa độ (r1, c1), (r2, c2).
        Trả về: True nếu nối được (<= 3 đường thẳng), ngược lại False.
        """
        # Lưu ý: Phần này bạn import hoặc bê hàm check_match(p1, p2) từ module game gốc sang.
        # Ví dụ giả định: return game_logic.check_match(self.state, p1, p2)
        pass

    def play_step(self, action):
        """
        Nhiệm vụ 1 & 3: Nhận tọa độ từ AI, thực thi logic, loại bỏ pygame.event.get() 
        và trả về bộ 4 biến (state, reward, done, score).
        
        @param action: Tuple chứa tọa độ của 2 ô AI muốn chọn: ((r1, c1), (r2, c2))
        """
        p1, p2 = action
        r1, c1 = p1
        r2, c2 = p2
        
        reward = 0
        done = False
        
        # BỎ QUA HOÀN TOÀN pygame.event.get() -> AI truyền action trực tiếp vào đây.
        
        # 1. Kiểm tra tính hợp lệ của Action
        # Không được chọn cùng 1 ô, không chọn ô trống, và 2 ô phải cùng loại Pokemon
        if p1 == p2 or self.state[r1][c1] == 0 or self.state[r2][c2] == 0 or self.state[r1][c1] != self.state[r2][c2]:
            reward = -2  # Phạt nặng nếu AI chọn bừa ô không hợp lệ hoặc sai cặp
            return self.state, reward, done, self.score

        # 2. Gọi thuật toán kiểm tra đường nối (Nội tại của game)
        is_valid_path = self._check_path_internal(p1, p2)
        
        if is_valid_path:
            # Nếu nối thành công: Xóa 2 ô trên bàn cờ (biến thành số 0)
            self.state[r1][c1] = 0
            self.state[r2][c2] = 0
            
            self.score += 10    # Tăng điểm số trong game
            reward = +10        # Thưởng điểm cho AI
            
            # 3. Kiểm tra xem đã phá đảo chưa (Done)
            # Nếu toàn bộ vùng lõi đều bằng 0 -> Hết sạch Pokemon
            if np.sum(self.state[1:self.rows+1, 1:self.cols+1]) == 0:
                reward = +50    # Thưởng lớn khi thắng game
                done = True
        else:
            # Chọn đúng cặp giống nhau nhưng đường đi bị chặn bởi ô khác
            reward = -1 
            
        # Thêm cơ chế kiểm tra bế tắc (Kịch đường đi): 
        # Nếu bàn cờ còn hình nhưng không còn cặp nào nối được, cũng kết thúc (done = True, reward = -5)
        
        return self.state, reward, done, self.score