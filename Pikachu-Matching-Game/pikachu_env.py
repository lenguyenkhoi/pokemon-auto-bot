import os
import random
import sys
import collections
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import pygame
except ImportError:  # pragma: no cover - môi trường không có pygame
    pygame = None

# Import trực tiếp các hàm tiện ích từ file của Người số 3 gửi
from utils.state_utils import (
    board_to_numpy, get_valid_actions, is_board_empty,
    alterBoardWithLevel, getRandomizedBoard,
)

class PikachuEnv:
    def __init__(self, board_width=14, board_height=9, level=1, render_mode=False):
        """
        Khởi tạo môi trường game Pikachu đồng bộ với thiết kế của Người thứ 3.
        """
        self.BOARDWIDTH = board_width
        self.BOARDHEIGHT = board_height
        self.LEVEL = level
        self.render_mode = render_mode
        
        self.BOXSIZE = 66
        self.WINDOWWIDTH = 1214
        self.WINDOWHEIGHT = 680
        self.NUMSAMEHEROES = 4
        
        self.XMARGIN = (self.WINDOWWIDTH - (self.BOXSIZE * self.BOARDWIDTH)) // 2
        self.YMARGIN = (self.WINDOWHEIGHT - (self.BOXSIZE * self.BOARDHEIGHT)) // 2
        
        if pygame is None:
            raise ImportError("pygame is required to use PikachuEnv")

        pygame.init()
        if self.render_mode:
            self.DISPLAYSURF = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
            pygame.display.set_caption('Pikachu AI Environment')
        else:
            self.DISPLAYSURF = pygame.Surface((self.WINDOWWIDTH, self.WINDOWHEIGHT))
            
        self.reset()

    def reset(self):
        """Khởi động lại ván game mới sử dụng hàm sinh map chuẩn của nhóm"""
        # Sử dụng hàm sinh map ngẫu nhiên mà Người số 3 đã đồng bộ từ mã nguồn gốc
        self.mainBoard = getRandomizedBoard(
            board_width=self.BOARDWIDTH,
            board_height=self.BOARDHEIGHT,
            num_same_heroes=self.NUMSAMEHEROES,
        )
        self.score = 0
        self.frame_iteration = 0
        self.game_over = False
        
        # Trả về ma trận dưới dạng numpy array chuẩn hóa
        return board_to_numpy(self.mainBoard)

    def play_step(self, action):
        """
        AI thực hiện một hành động dựa trên cấu trúc phẳng của Người số 3.
        action: tuple phẳng (r1, c1, r2, c2) tức là (y1, x1, y2, x2)
        """
        self.frame_iteration += 1
        reward = 0
        
        # 1. Khống chế phạt nếu AI bị kẹt (quá nhiều bước không ăn được điểm)
        max_frames = (self.BOARDWIDTH * self.BOARDHEIGHT) * 4
        if self.frame_iteration > max_frames:
            self.game_over = True
            reward = -10  
            return np.array(self.mainBoard, dtype=np.int32), reward, self.game_over, self.score

        # Giải nén hành động theo đúng thứ tự (Row1, Col1, Row2, Col2) của Người 3
        r1, c1, r2, c2 = action

        # Kiểm tra tọa độ xem có nằm trong phạm vi ma trận không
        if not (0 <= r1 < self.BOARDHEIGHT and 0 <= c1 < self.BOARDWIDTH and
                0 <= r2 < self.BOARDHEIGHT and 0 <= c2 < self.BOARDWIDTH):
            reward = -1
            return np.array(self.mainBoard, dtype=np.int32), reward, self.game_over, self.score

        # 2. Kiểm tra tính hợp lệ bằng hàm BFS (Truyền self.mainBoard và tọa độ vào)
        if (self.mainBoard[r1][c1] != 0 and 
            self.mainBoard[r2][c2] != 0 and 
            (r1, c1) != (r2, c2) and 
            self.bfs(r1, c1, r2, c2)):
            
            # Nếu nối thành công: Xóa 2 ô (set về 0)
            self.mainBoard[r1][c1] = 0
            self.mainBoard[r2][c2] = 0
            
            # Áp dụng cơ chế dịch chuyển ô theo Level (Sử dụng hàm gộp từ Người 3)
            self.mainBoard = alterBoardWithLevel(self.mainBoard, r1, c1, r2, c2, self.LEVEL)
            
            self.score += 10
            reward = 20  
            self.frame_iteration = 0  
            
            # Kiểm tra sạch bàn bằng hàm của Người số 3
            if is_board_empty(self.mainBoard):
                self.game_over = True
                reward = 50  
        else:
            # Chọn sai hoặc chọn ô trống
            reward = -1  

        if self.render_mode:
            self.render()

        return np.array(self.mainBoard, dtype=np.int32), reward, self.game_over, self.score

    def bfs(self, boxy1, boxx1, boxy2, boxx2):
        """
        Thuật toán BFS gốc của bạn, được tinh chỉnh nhẹ để nhận tham số 
        hệ tọa độ (Row, Col) đồng bộ với cách gọi hàm của Người số 3.
        """
        if self.mainBoard[boxy1][boxx1] != self.mainBoard[boxy2][boxx2]:
            return []

        n = self.BOARDHEIGHT
        m = self.BOARDWIDTH

        q = collections.deque()
        q.append((boxy1, boxx1, 0, 'no_direction'))
        visited = set()
        visited.add((boxy1, boxx1, 0, 'no_direction'))

        while len(q) > 0:
            r, c, num_turns, direction = q.popleft()
            if (r, c) != (boxy1, boxx1) and (r, c) == (boxy2, boxx2):
                return [1]  # Trả về kết quả đại diện đường đi hợp lệ

            dict_directions = {(r + 1, c): 'down', (r - 1, c): 'up', (r, c - 1): 'left', (r, c + 1): 'right'}
            for neiborX, neiborY in dict_directions:
                next_direction = dict_directions[(neiborX, neiborY)]
                if 0 <= neiborX <= n - 1 and 0 <= neiborY <= m - 1 and (
                        self.mainBoard[neiborX][neiborY] == 0 or (neiborX, neiborY) == (boxy2, boxx2)):
                    if direction == 'no_direction':
                        q.append((neiborX, neiborY, num_turns, next_direction))
                        visited.add((neiborX, neiborY, num_turns, next_direction))
                    elif direction == next_direction and (neiborX, neiborY, num_turns, next_direction) not in visited:
                        q.append((neiborX, neiborY, num_turns, next_direction))
                        visited.add((neiborX, neiborY, num_turns, next_direction))
                    elif direction != next_direction and num_turns < 2 and (neiborX, neiborY, num_turns + 1, next_direction) not in visited:
                        q.append((neiborX, neiborY, num_turns + 1, next_direction))
                        visited.add((neiborX, neiborY, num_turns + 1, next_direction))
        return []

    def render(self):
        """Vẽ giao diện text ma trận số trực quan hiển thị trạng thái thực tế"""
        self.DISPLAYSURF.fill((60, 60, 100))
        font = pygame.font.SysFont('arial', 24)
        for boxy in range(self.BOARDHEIGHT):
            for boxx in range(self.BOARDWIDTH):
                val = self.mainBoard[boxy][boxx]
                if val != 0:
                    left = boxx * self.BOXSIZE + self.XMARGIN
                    top = boxy * self.BOXSIZE + self.YMARGIN
                    pygame.draw.rect(self.DISPLAYSURF, (255, 255, 255), (left, top, self.BOXSIZE-4, self.BOXSIZE-4))
                    text_surf = font.render(str(val), True, (0, 0, 0))
                    self.DISPLAYSURF.blit(text_surf, (left + 15, top + 15))
                    
        score_surf = font.render(f"AI Score: {self.score} | Steps: {self.frame_iteration}", True, (255, 255, 0))
        self.DISPLAYSURF.blit(score_surf, (20, 20))
        pygame.display.update()

# ==============================================================================
# ĐOẠN SCRIPT CHẠY THỬ PIPELINE KẾT HỢP GIỮA NGƯỜI SỐ 2 VÀ NGƯỜI SỐ 3
# ==============================================================================
if __name__ == '__main__':
    # Tạo môi trường giả lập hiển thị
    env = PikachuEnv(board_width=14, board_height=9, level=1, render_mode=True)
    board_state = env.reset()
    
    print("--- KẾT NỐI PIPELINE GIỮA NGƯỜI 2 & NGƯỜI 3 THÀNH CÔNG ---")
    
    clock = pygame.time.Clock()
    for _ in range(5):
        # Người số 3 cung cấp hàm tìm tất cả các action hợp lệ thật sự trên bàn cờ
        # Truyền chính hàm env.bfs của bạn vào để test độ khớp logic
        valid_moves = get_valid_actions(board_state, env.bfs)
        
        if valid_moves:
            # Giả lập AI bốc bừa một nước đi đúng 100% trong danh sách hợp lệ
            chosen_action = random.choice(valid_moves)
            r1, c1, r2, c2 = chosen_action
            print(f"AI chọn cặp hợp lệ: Ô ({r1},{c1}) nối với ({r2},{c2})")
            
            # Đẩy action chuẩn của Người 3 vào hàm play_step của bạn
            board_state, reward, done, score = env.play_step(chosen_action)
            print(f"Kết quả -> Reward nhận được: {reward} | Điểm số hiện tại: {score}")
        else:
            print("Bàn cờ bị deadlock (không còn nước đi hợp lệ)!")
            board_state = env.reset()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        clock.tick(1) # Chạy chậm 1 giây/bước để dễ theo dõi log terminal
    pygame.quit()