# config.py

# Toạ độ vùng chơi Pokémon 
BOARD_REGION = {
    "top": 151,
    "left": 46,
    "width": 1691,
    "height": 952
}

# Cấu hình ma trận (9 hàng, 16 cột)
ROWS = 9
COLS = 16

# Kích thước pixel của 1 ô Pokémon
CELL_WIDTH = BOARD_REGION["width"] // COLS
CELL_HEIGHT = BOARD_REGION["height"] // ROWS

# Toạ độ điểm click điều khiển các nút chức năng ở rìa bên phải
RESTART_BUTTON_POS = (1820, 435)  # Vị trí nút mũi tên xoay vòng (Xáo bài/Chơi lại)
SHUFFLE_BUTTON_POS = (1820, 340)  # Vị trí nút có hình dấu hỏi chấm (Gợi ý)

# Đường dẫn tài nguyên và số lượng hình pokemon
TEMPLATES_DIR = "assets/"
NUM_POKEMON_TYPES = 18