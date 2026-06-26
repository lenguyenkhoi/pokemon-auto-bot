import pygame
import random
import numpy as np

class PokemonEnv:
    def __init__(self, width=16, height=9):
        self.width = width
        self.height = height
        # Định nghĩa các phần thưởng (Reward)
        self.REWARD_MATCH = 10      # Nối đúng cặp
        self.REWARD_WRONG = -2      # Chọn sai hoặc không nối được
        self.REWARD_WIN = 50        # Thắng game (hết map)
        self.REWARD_STALL = -1      # Phạt mỗi bước đi quá lâu hoặc chọn lại ô cũ
        
        self.reset()

    def reset(self):
        """
        Khởi tạo lại map mới ngẫu nhiên mỗi vòng chơi.
        Đảm bảo các icon xuất hiện theo cặp để game có thể giải được.
        """
        self.score = 0
        self.done = False
        self.selected_tile = None # Lưu tọa độ ô thứ nhất được chọn (x1, y1)
        
        # 1. Tạo danh sách các cặp icon ngẫu nhiên (ví dụ có 20 loại pokemon)
        num_tiles = self.width * self.height
        assert num_tiles % 2 == 0, "Tổng số ô của map phải là số chẵn!"
        
        half_tiles = num_tiles // 2
        # Tạo danh sách các id icon từ 1 đến 20
        pool = [random.randint(1, 20) for _ in range(half_tiles)]
        # Nhân đôi để tạo thành cặp
        pool = pool + pool
        # Trộn ngẫu nhiên
        random.shuffle(pool)
        
        # 2. Đưa vào ma trận map (0 đại diện cho ô trống đã được xóa)
        self.matrix = np.array(pool).reshape((self.height, self.width))
        
        # Trả về trạng thái ban đầu
        return self._get_state()

    def play_step(self, action):
        """
        Nhận tọa độ click từ AI, thực thi và trả về (state, reward, done, score)
        action: tuple (x, y) - tọa độ ô mà AI muốn click
        """
        x, y = action
        reward = self.REWARD_STALL
        
        # Kiểm tra click hợp lệ (nằm trong map và ô đó không trống)
        if not (0 <= x < self.width and 0 <= y < self.height) or self.matrix[y][x] == 0:
            return self._get_state(), self.REWARD_WRONG, self.done, self.score

        # Nếu chưa chọn ô thứ nhất
        if self.selected_tile is None:
            self.selected_tile = (x, y)
            reward = 0 # Click ô đầu tiên chưa tính điểm phạt/thưởng lớn
        else:
            x1, y1 = self.selected_tile
            
            # Click trùng lại ô cũ
            if (x1, y1) == (x, y):
                self.selected_tile = None
                reward = self.REWARD_WRONG
            else:
                # 2. Gọi thuật toán check đường dẫn nội tại của game (đường thẳng, 1 góc vuông, 2 góc vuông)
                if self.matrix[y1][x1] == self.matrix[y][x] and self._check_path(x1, y1, x, y):
                    # Nếu nối thành công: Xóa 2 ô khỏi map
                    self.matrix[y1][x1] = 0
                    self.matrix[y][x] = 0
                    reward = self.REWARD_MATCH
                    self.score += 10
                    
                    # Kiểm tra xem đã hết map chưa (Thắng)
                    if np.all(self.matrix == 0):
                        reward += self.REWARD_WIN
                        self.done = True
                else:
                    # Chọn sai cặp hoặc không có đường nối
                    reward = self.REWARD_WRONG
                
                # Reset trạng thái chọn sau lượt click thứ 2
                self.selected_tile = None

        # Trả về bộ 4 biến chuẩn RL
        return self._get_state(), reward, self.done, self.score

    def _get_state(self):
        """
        Trả về trạng thái hiện tại của môi trường dưới dạng ma trận.
        Có thể flat ra hoặc giữ nguyên để AI CNN xử lý.
        """
        # Trả về map hiện tại + ô đang được chọn (nếu có) để AI biết nó đang chọn gì
        selected_mask = np.zeros_like(self.matrix)
        if self.selected_tile:
            x, y = self.selected_tile
            selected_mask[y][x] = 1
            
        return np.stack([self.matrix, selected_mask], axis=-1)

    def _check_path(self, x1, y1, x2, y2):
        """
        Hàm nội tại kiểm tra đường đi (Thuật toán tìm đường Pikachu của bạn).
        Thay thế đoạn này bằng logic tìm đường nối (tối đa 3 đoạn thẳng) sẵn có của bạn.
        """
        # Giả lập logic kiểm tra: trả về True nếu thông đường, False nếu bị chặn
        # BẠN CẦN CHÈN THUẬT TOÁN LOGIC GAME CỦA BẠN VÀO ĐÂY
        return True 

    def render(self, screen):
        """
        Hàm vẽ giao diện (chỉ dùng để xem, KHÔNG dùng để nhận sự kiện)
        """
        screen.fill((255, 255, 255))
        # Logic vẽ các ô pokemon dựa trên self.matrix lên màn hình pygame...
        pygame.display.flip()