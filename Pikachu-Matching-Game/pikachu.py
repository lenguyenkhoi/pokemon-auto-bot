import pygame, sys, random, copy, time, collections, os, json

# Ensure working directory is the script directory so relative resource
# paths like 'Resources/...' resolve correctly when running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
from pygame.locals import *
from datetime import datetime
import numpy as np
import scipy.ndimage

# AI Autopilot Configuration
AUTOPILOT = False
ai_model = None
USER = None
USER_NAME = ""

if "--ai" in sys.argv:
    AUTOPILOT = True
    USER = {"Board": [[0]*14 for _ in range(9)], "Score": 0, "Level": 1}
    USER_NAME = "ai_agent"
    
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
        
    import yaml
    import torch
    from model.model import PokemonModel
    from utils.state_utils import extract_state, state_to_torch, get_valid_actions, qvalues_to_action
    
    # Load config and weights from parent directory
    with open("../configs/agent_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    output_size = 14 * 9
    ai_model = PokemonModel(config, output_size)
    model_path = "../weights/best_model.pt"
    if os.path.exists(model_path):
        print(f"Loading AI model weights from {model_path}...")
        ai_model.load(model_path, map_location="cpu")
    else:
        print(f"WARNING: Weights {model_path} not found. Running with initial random weights.")
    ai_model.eval()

"""Đặt các cấu hình mặc định"""
FPS = 60
WINDOWWIDTH = 1214
WINDOWHEIGHT = 680
BOXSIZE = 66
BOARDWIDTH = 14
BOARDHEIGHT = 9
NUMHEROES_ONBOARD = 21
NUMSAMEHEROES = 4
TIMEBAR_LENGTH = 300
TIMEBAR_WIDTH = 30
LEVELMAX = 5
LIVES = 10
GAMETIME = 360
GETHINTTIME = 20
SOUND_ON = True 
LEVEL = 1 
XMARGIN = (WINDOWWIDTH - (BOXSIZE * BOARDWIDTH)) // 2
YMARGIN = (WINDOWHEIGHT - (BOXSIZE * BOARDHEIGHT)) // 2
# set up the colors
GRAY = (100, 100, 100)
NAVYBLUE = ( 60, 60, 100)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = ( 0, 255, 0)
BOLDGREEN = (0, 175, 0)
BLUE = ( 0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 128, 0)
PURPLE = (255, 0, 255)
CYAN = ( 0, 255, 255)
BLACK = (0, 0, 0)
BGCOLOR = NAVYBLUE
HIGHLIGHTCOLOR = BLUE
BORDERCOLOR = RED
SKIN_COLOR = (255, 228, 196)
# TIMEBAR setup
barPos = (WINDOWWIDTH // 2 - TIMEBAR_LENGTH // 2, YMARGIN // 2 - TIMEBAR_WIDTH // 2)
barSize = (TIMEBAR_LENGTH, TIMEBAR_WIDTH)
borderColor = WHITE
barColor = BOLDGREEN
ACCOUNTS_FILE = "User_data/accounts.json"
# Load pictures
aegis = pygame.image.load('Resources/others/aegis_2.jpg')
aegis = pygame.transform.scale(aegis, (45, 45))
pygame.font.init()
# Load background
startBG = pygame.image.load('Resources/Background/startBG.png')
startBG = pygame.transform.scale(startBG, (WINDOWWIDTH, WINDOWHEIGHT))
BG = pygame.image.load('Resources/Background/main.png')
BG = pygame.transform.scale(BG, (WINDOWWIDTH, WINDOWHEIGHT))
# Load sound and musicbbb
pygame.mixer.pre_init()
pygame.mixer.init()
clickSound = pygame.mixer.Sound('Resources/sound_effect/click_selecting.mp3')
getPointSound = pygame.mixer.Sound('Resources/sound_effect/victory.mp3')
WrongSound = pygame.mixer.Sound('Resources/sound_effect/wrong.mp3')
startScreenSound = pygame.mixer.Sound('Resources/sound_effect/warriors-of-the-night-assemble.wav')
listMusicBG = ["Resources/BGmusic/" + i for i in os.listdir("Resources/BGmusic")]
font_path = "Resources/Fonts/Jersey15-Regular.ttf"
click = pygame.mixer.Sound("Resources/sound_effect/mouse_click.mp3")

"""Xử lý đăng kí, đăng nhập tài khoản"""
def mouseclick():
    click.play()

def showLoginScreen():
    """Hiển thị màn hình đăng nhập"""
    login_font = pygame.font.Font(font_path, 50)
    input_font = pygame.font.Font(font_path, 40)
    button_font = pygame.font.Font(font_path, 45)
    username = ""
    password = ""
    active_field = "username"
    error_message = ""
    
    # Kích thước và tọa độ các thành phần
    field_width = 400
    field_height = 50
    label_offset_x = 80
    text_offset_y = -2
    sigma = 5
    startBG = pygame.image.load("Resources/Background/startBG.png")
    startBG = pygame.transform.scale(startBG, (WINDOWWIDTH, WINDOWHEIGHT))
    arr = pygame.surfarray.array3d(startBG)  # Chuyển thành numpy array
    blurred_arr = scipy.ndimage.gaussian_filter(arr, sigma=(sigma, sigma, 0))  # Làm mờ Gaussian
    blurred_bg =  pygame.surfarray.make_surface(blurred_arr)
    login_button = pygame.image.load("Resources/Buttons/Blank/A Buttons Medium1.png")
    login_button_hover = pygame.image.load("Resources/Buttons/Blank/B Buttons Medium2.png")
    login_button = pygame.transform.scale(login_button,(240, 80))
    login_button_hover = pygame.transform.scale(login_button_hover,(240, 80))
    login_button_rect = login_button.get_rect()
    login_button_rect.center = (WINDOWWIDTH // 2 , WINDOWHEIGHT//2 + 100)
    button_text = login_font.render("LOGIN", True, (255,255,0))
    button_rect = button_text.get_rect(center = (WINDOWWIDTH // 2 , WINDOWHEIGHT//2 + 100) )
    login_title = pygame.image.load("Resources/others/login2.png")
    login_title_rect = login_title.get_rect()
    login_title_rect.center = (WINDOWWIDTH//2, 80)
    board = pygame.image.load("Resources/others/Board.png")
    board = pygame.transform.scale(board,(720,660))
    board_rect = board.get_rect()
    board_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT//2)
    back = pygame.image.load("Resources/Buttons/Back2/A_Back1.png")
    back = pygame.transform.scale(back,(240,80))
    back_hover = pygame.image.load("Resources/Buttons/Back2/B_Back1.png")
    back_hover = pygame.transform.scale(back_hover,(240,80))
    back_rect = back.get_rect()
    back_rect.center = (120,60)
    while True:
        # Vẽ nền
        DISPLAYSURF.blit(blurred_bg, (0,0))
        DISPLAYSURF.blit(board,board_rect)

        # Tiêu đề
        DISPLAYSURF.blit(login_title, login_title_rect)

        # Nhãn và trường nhập
        username_label = input_font.render("Username:", True, (255, 255, 0))  # Chữ đỏ
        username_label_rect = username_label.get_rect(midright=(WINDOWWIDTH // 2 - field_width // 2 - label_offset_x + 140, 200 + field_height // 2))
        username_rect = pygame.Rect(WINDOWWIDTH // 2 - field_width // 2 + 90, 200, field_width, field_height)

        password_label = input_font.render("Password:", True, (255, 255, 0))  # Chữ đỏ
        password_label_rect = password_label.get_rect(midright=(WINDOWWIDTH // 2 - field_width // 2 - label_offset_x + 140, 300 + field_height // 2))
        password_rect = pygame.Rect(WINDOWWIDTH // 2 - field_width // 2 + 90, 300, field_width, field_height)

        # Vẽ nhãn và trường nhập
        DISPLAYSURF.blit(username_label, username_label_rect)
        pygame.draw.rect(DISPLAYSURF, WHITE, username_rect)
        pygame.draw.rect(DISPLAYSURF, BLACK, username_rect, 2)
        DISPLAYSURF.blit(input_font.render(username, True, BLACK), (username_rect.x + 10, username_rect.y + text_offset_y))

        DISPLAYSURF.blit(password_label, password_label_rect)
        pygame.draw.rect(DISPLAYSURF, WHITE, password_rect)
        pygame.draw.rect(DISPLAYSURF, BLACK, password_rect, 2)
        DISPLAYSURF.blit(input_font.render("*" * len(password), True, BLACK), (password_rect.x + 10, password_rect.y + text_offset_y))

        # Nút back
        DISPLAYSURF.blit(back, back_rect)
        # Nút LOGIN
        DISPLAYSURF.blit(login_button, login_button_rect)
        DISPLAYSURF.blit(button_text, button_rect)
        if login_button_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(login_button_hover, login_button_rect)
            DISPLAYSURF.blit(button_text, button_rect)
        if back_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(back_hover, back_rect)
        # Thông báo lỗi (nếu có)
        if error_message:
            error_text = input_font.render(error_message, True, YELLOW)
            error_rect = error_text.get_rect(center=(WINDOWWIDTH // 2, 500))
            DISPLAYSURF.blit(error_text, error_rect)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if active_field == "username":
                    if event.key == pygame.K_BACKSPACE:
                        username = username[:-1]
                    else:
                        username += event.unicode
                elif active_field == "password":
                    if event.key == pygame.K_BACKSPACE:
                        password = password[:-1]
                    else:
                        password += event.unicode
            elif event.type == MOUSEBUTTONDOWN:
                mousex, mousey = event.pos
                if username_rect.collidepoint(mousex, mousey):
                    active_field = "username"
                elif password_rect.collidepoint(mousex, mousey):
                    active_field = "password"
                elif login_button_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    if check_login(username, password):
                        return False
                    else:
                        error_message = "Invalid username or password"
                elif back_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    return True

def check_login(username, password):
    """Kiểm tra thông tin đăng nhập"""
    global USER, USER_NAME
    try:
        with open(ACCOUNTS_FILE, "r") as f:
            accounts = json.load(f)

        # Kiểm tra xem `username` có tồn tại trong `accounts` không
        user_data = accounts.get(username.lower())
        if user_data and user_data.get("password") == str(len(username)) + username.lower() + str(len(password)) + password.lower():
            USER = user_data
            USER_NAME = username.lower()
            return True
        else:
            return False
    except FileNotFoundError:
        # File không tồn tại
        print("Error: Accounts file not found.")
        return False
    except json.JSONDecodeError:
        # File JSON không hợp lệ
        print("Error: Invalid JSON format in accounts file.")
        return False
    except Exception as e:
        # Xử lý các lỗi không mong đợi khác
        print(f"Unexpected error: {e}")
        return False

def register_account(username, password):
    """Xử lý đăng ký tài khoản"""
    global USER, USER_NAME
    try:
        with open(ACCOUNTS_FILE, "r") as f:
            accounts = json.load(f)
    except FileNotFoundError:
        accounts = {}

    if username.lower() in accounts:
        return False  # Tên tài khoản đã tồn tại
    accounts[username.lower()] = {"password":str(len(username)) + username.lower() + str(len(password)) + password.lower(), "Board":getRandomizedBoard(), "Score": 0, "Level": 1}
    USER = accounts[username.lower()]
    USER_NAME = username.lower()
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(accounts, f, indent = 4)
    return True

def showRegisterScreen():
    """Màn hình đăng kí"""
    login_font = pygame.font.Font('Resources/Fonts/Jersey15-Regular.ttf', 50)
    input_font = pygame.font.Font('Resources/Fonts/Jersey15-Regular.ttf', 40)
    button_font = pygame.font.Font('Resources/Fonts/Jersey15-Regular.ttf', 45)
    username = ""
    password = ""
    active_field = "username"
    message = ""

    # Kích thước và tọa độ các thành phần
    field_width = 400
    field_height = 50
    label_offset_x = 80  # Khoảng cách giữa nhãn và khung nhập
    text_offset_y = -5  # Điều chỉnh nhích chữ lên trên khung nhập
    sigma = 5
    startBG = pygame.image.load("Resources/Background/startBG.png")
    startBG = pygame.transform.scale(startBG, (WINDOWWIDTH, WINDOWHEIGHT))
    arr = pygame.surfarray.array3d(startBG)  # Chuyển thành numpy array
    blurred_arr = scipy.ndimage.gaussian_filter(arr, sigma=(sigma, sigma, 0))  # Làm mờ Gaussian
    blurred_bg =  pygame.surfarray.make_surface(blurred_arr)
    register_button = pygame.image.load("Resources/Buttons/Blank/A Buttons Medium1.png")
    register_button_hover = pygame.image.load("Resources/Buttons/Blank/B Buttons Medium2.png")
    register_button = pygame.transform.scale(register_button,(240, 80))
    register_button_hover = pygame.transform.scale(register_button_hover,(240, 80))
    register_button_rect = register_button.get_rect()
    register_button_rect.center = (WINDOWWIDTH // 2 , WINDOWHEIGHT//2 + 100)
    button_text = login_font.render("REGISTER", True, (255,255,255))
    button_rect = button_text.get_rect(center = (WINDOWWIDTH // 2 , WINDOWHEIGHT//2 + 100) )
    register_title = pygame.image.load("Resources/others/register.png")
    register_title_rect = register_title.get_rect()
    register_title_rect.center = (WINDOWWIDTH//2, 80)
    board = pygame.image.load("Resources/others/Board.png")
    board = pygame.transform.scale(board,(720,660))
    board_rect = board.get_rect()
    board_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT//2)
    back = pygame.image.load("Resources/Buttons/Back2/A_Back1.png")
    back = pygame.transform.scale(back,(240,80))
    back_hover = pygame.image.load("Resources/Buttons/Back2/B_Back1.png")
    back_hover = pygame.transform.scale(back_hover,(240,80))
    back_rect = back.get_rect()
    back_rect.center = (120,60)

    while True:

        DISPLAYSURF.blit(blurred_bg, (0,0))
        DISPLAYSURF.blit(board, board_rect)
        DISPLAYSURF.blit(register_title, register_title_rect)
        DISPLAYSURF.blit(register_button, register_button_rect)
        DISPLAYSURF.blit(button_text, button_rect)
        DISPLAYSURF.blit(back, back_rect)
        if register_button_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(register_button_hover, register_button_rect)
            DISPLAYSURF.blit(button_text, button_rect)
        if back_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(back_hover, back_rect)

        # Nhãn và trường nhập cho Username
        username_label = input_font.render("Username:", True, YELLOW)
        username_label_rect = username_label.get_rect(midright=(WINDOWWIDTH // 2 - field_width // 2 - label_offset_x + 140, 200 + field_height // 2))
        username_rect = pygame.Rect(WINDOWWIDTH // 2 - field_width // 2 + 90, 200, field_width, field_height)

        # Nhãn và trường nhập cho Password
        password_label = input_font.render("Password:", True, YELLOW)
        password_label_rect = password_label.get_rect(midright=(WINDOWWIDTH // 2 - field_width // 2 - label_offset_x + 140, 300 + field_height // 2))
        password_rect = pygame.Rect(WINDOWWIDTH // 2 - field_width // 2 + 90, 300, field_width, field_height)

        # Vẽ nhãn và trường nhập
        DISPLAYSURF.blit(username_label, username_label_rect)
        pygame.draw.rect(DISPLAYSURF, WHITE, username_rect)
        pygame.draw.rect(DISPLAYSURF, BLACK, username_rect, 2)
        DISPLAYSURF.blit(input_font.render(username, True, BLACK), (username_rect.x + 10, username_rect.y + text_offset_y))

        DISPLAYSURF.blit(password_label, password_label_rect)
        pygame.draw.rect(DISPLAYSURF, WHITE, password_rect)
        pygame.draw.rect(DISPLAYSURF, BLACK, password_rect, 2)
        DISPLAYSURF.blit(input_font.render("*" * len(password), True, BLACK), (password_rect.x + 10, password_rect.y + text_offset_y))

        # Thông báo (nếu có)
        if message:
            message_text = input_font.render(message, True, YELLOW)
            message_rect = message_text.get_rect(center=(WINDOWWIDTH // 2, 500))
            DISPLAYSURF.blit(message_text, message_rect)

        pygame.display.update()

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if active_field == "username":
                    if event.key == pygame.K_BACKSPACE:
                        username = username[:-1]
                    else:
                        username += event.unicode
                elif active_field == "password":
                    if event.key == pygame.K_BACKSPACE:
                        password = password[:-1]
                    else:
                        password += event.unicode
            elif event.type == MOUSEBUTTONDOWN:
                mousex, mousey = event.pos
                if username_rect.collidepoint(mousex, mousey):
                    active_field = "username"
                elif password_rect.collidepoint(mousex, mousey):
                    active_field = "password"
                elif register_button_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    if register_account(username, password):
                        message = "Registration successful!"
                        return False
                    else:
                        message = "Username already exists!"
                elif back_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    return True

# Hàm vẽ màn hình đăng nhập
def showMainAuthScreen():
    """Hiển thị màn hình đăng nhập, đăng ký"""
    login_font = pygame.font.Font('Resources/Fonts/Jersey15-Regular.ttf', 45)
    button_font = pygame.font.Font('Resources/Fonts/Jersey15-Regular.ttf', 45)
    login_button = pygame.image.load("Resources/Buttons/Blank/A Buttons Medium1.png")
    login_button = pygame.transform.scale(login_button,(240,80))
    login_button_hover = pygame.image.load("Resources/Buttons/Blank/B Buttons Medium2.png")
    login_button_hover = pygame.transform.scale(login_button_hover,(240,80))
    login_button_rect = login_button.get_rect()
    login_button_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT // 2 - 60)
    login_text = button_font.render("LOGIN", True, YELLOW)
    login_text_rect = login_text.get_rect(center = (WINDOWWIDTH//2, WINDOWHEIGHT // 2 - 60) )
    register_button = pygame.image.load("Resources/Buttons/Blank/A Buttons Medium1.png")
    register_button = pygame.transform.scale(register_button,(240,80))
    register_button_hover = pygame.image.load("Resources/Buttons/Blank/B Buttons Medium2.png")
    register_button_hover = pygame.transform.scale(register_button_hover,(240,80))
    register_button_rect = register_button.get_rect()
    register_button_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT // 2 + 60)
    register_text = button_font.render("REGISTER", True, WHITE)
    register_text_rect = register_text.get_rect(center = (WINDOWWIDTH//2, WINDOWHEIGHT // 2 + 60) )
    sigma = 5
    startBG = pygame.image.load("Resources/Background/startBG.png")
    startBG = pygame.transform.scale(startBG, (WINDOWWIDTH, WINDOWHEIGHT))
    arr = pygame.surfarray.array3d(startBG)  # Chuyển thành numpy array
    blurred_arr = scipy.ndimage.gaussian_filter(arr, sigma=(sigma, sigma, 0))  # Làm mờ Gaussian
    blurred_bg =  pygame.surfarray.make_surface(blurred_arr)
    board = pygame.image.load("Resources/others/Board.png")
    board = pygame.transform.scale(board,(600,550))
    board_rect = board.get_rect()
    board_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT//2)
    welcome = pygame.image.load("Resources/others/welcome.png")
    welcome_rect = welcome.get_rect(center = (WINDOWWIDTH//2, 125))
    while True:
        #Hiện background
        DISPLAYSURF.blit(blurred_bg, (0,0))

        #Hiện bảng
        DISPLAYSURF.blit(board, board_rect)

        #Hiện tiêu đề
        DISPLAYSURF.blit(welcome, welcome_rect)

        #Hiện các nút
        DISPLAYSURF.blit(login_button, login_button_rect)
        DISPLAYSURF.blit(login_text, login_text_rect)
        DISPLAYSURF.blit(register_button, register_button_rect)
        DISPLAYSURF.blit(register_text, register_text_rect)
        if login_button_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(login_button_hover, login_button_rect)
            DISPLAYSURF.blit(login_text, login_text_rect)
        
        if register_button_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(register_button_hover, register_button_rect)
            DISPLAYSURF.blit(register_text, register_text_rect)

        pygame.display.update()

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                mousex, mousey = event.pos
                if login_button_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    return showLoginScreen()  # Hiển thị màn hình đăng nhập
                elif register_button_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    return showRegisterScreen()  # Hiển thị màn hình đăng ký

"""Màn hình khởi đầu và các nút"""
def showStartScreen():
    Path = "Resources/Buttons/"
    newGame1 = pygame.image.load( Path + "NewGame/A_Newgame1.png")
    newGame2 = pygame.image.load( Path + "NewGame/B_NewGame1.png")
    newGame1 = pygame.transform.scale(newGame1, (192,64))
    newGame2 = pygame.transform.scale(newGame2, (192,64))
    newGameRect = newGame1.get_rect()
    newGameRect.center = (WINDOWWIDTH // 4 + 20, 4 * WINDOWHEIGHT // 9 + 80)
    Continue1 = pygame.image.load(Path + "Continue/A_Continue1.png")
    Continue2 = pygame.image.load(Path + "Continue/B_Continue1.png")
    Continue1 = pygame.transform.scale(Continue1, (192,64))
    Continue2 = pygame.transform.scale(Continue2, (192,64))
    ContinueRect = Continue1.get_rect()
    ContinueRect.center = (WINDOWWIDTH // 4 + 20, 4 * WINDOWHEIGHT // 9 )
    Setting1 = pygame.image.load( Path + "Settings/A_Settings1.png")
    Setting2 = pygame.image.load( Path + "Settings/B_Settings1.png")
    Setting1 = pygame.transform.scale(Setting1, (192,64))
    Setting2 = pygame.transform.scale(Setting2, (192,64))
    SettingRect = Setting1.get_rect()
    SettingRect.center = (WINDOWWIDTH // 4 + 20, 4 * WINDOWHEIGHT // 9 + 160)
    Exit1 = pygame.image.load( Path + "Exit/A_Exit1.png")
    Exit2 = pygame.image.load( Path + "Exit/B_Exit1.png")
    Exit1 = pygame.transform.scale(Exit1, (192,64))
    Exit2 = pygame.transform.scale(Exit2, (192,64))
    exitRect = Exit1.get_rect()
    exitRect.center = (WINDOWWIDTH // 4 + 20, 4 * WINDOWHEIGHT // 9 + 240)
    BTrophy = pygame.image.load("Resources/others/Trophies/bronze_trophy.png")
    STrophy = pygame.image.load("Resources/others/Trophies/silver_trophy.png")
    GTrophy = pygame.image.load("Resources/others/Trophies/gold_trophy.png")
    Leaderboard1 = pygame.image.load(Path + "Blank/A Buttons Medium1.png")
    Leaderboard2 = pygame.image.load(Path + "Blank/B Buttons Medium2.png")
    Leaderboard1 = pygame.transform.scale(Leaderboard1, (192,64))
    Leaderboard2 = pygame.transform.scale(Leaderboard2, (192,64))
    leaderboardRect = Leaderboard1.get_rect()
    leaderboardRect.center = ( 100, WINDOWHEIGHT - 50)
    STARTFONT = pygame.font.Font("Resources/Fonts/Jersey15-Regular.ttf", 36)
    logo = pygame.image.load("Resources/others/logo.png")
    startScreenSound.play()
    while True:
        DISPLAYSURF.blit(startBG, (0, 0))
        DISPLAYSURF.blit(logo, (WINDOWWIDTH//2 - 600, WINDOWHEIGHT // 10 - 60))
        DISPLAYSURF.blit(newGame1, newGameRect)
        DISPLAYSURF.blit(Continue1, ContinueRect)
        DISPLAYSURF.blit(Setting1, SettingRect)
        DISPLAYSURF.blit(Exit1, exitRect)
        DISPLAYSURF.blit(Leaderboard1, leaderboardRect)
        DISPLAYSURF.blit(BTrophy, ( 40, WINDOWHEIGHT - 66))
        DISPLAYSURF.blit(STrophy, (85, WINDOWHEIGHT - 66))
        DISPLAYSURF.blit(GTrophy, (130, WINDOWHEIGHT - 66))
        if newGameRect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(newGame2,newGameRect)
        if ContinueRect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(Continue2, ContinueRect)
        if SettingRect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(Setting2, SettingRect)
        if exitRect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(Exit2, exitRect)
        if leaderboardRect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(Leaderboard2, leaderboardRect)
            DISPLAYSURF.blit(BTrophy, ( 40, WINDOWHEIGHT - 66))
            DISPLAYSURF.blit(STrophy, (85, WINDOWHEIGHT - 66))
            DISPLAYSURF.blit(GTrophy, (130, WINDOWHEIGHT - 66))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEMOTION:
                mousex, mousey = event.pos
            elif event.type == MOUSEBUTTONDOWN:
                mousex, mousey = event.pos
                if newGameRect.collidepoint((mousex, mousey)):
                    mouseclick()
                    return "Newgame"
                elif SettingRect.collidepoint((mousex, mousey)):
                    mouseclick()
                    return "Setting"
                elif ContinueRect.collidepoint((mousex, mousey)):
                    mouseclick()
                    return "Continue"
                elif exitRect.collidepoint((mousex, mousey)):
                    mouseclick()
                    pygame.quit()
                    sys.exit()  # Quit the application
                    # Xử lý click vào nút "LEADERBOARD"
                elif leaderboardRect.collidepoint((mousex, mousey)):
                    mouseclick()
                    showLeaderboard(DISPLAYSURF, BASICFONT, WINDOWWIDTH, WINDOWHEIGHT)
                    return
        pygame.display.update()
        FPSCLOCK.tick(FPS)

"""Bảng Setting"""

"""Các nút"""
def get_buttons(directory, position,scale):
    List_dir = os.listdir(directory)
    unscale_buttons = [pygame.image.load(directory + "/" + i) for i in List_dir]
    buttons = [pygame.transform.scale(button, scale) for button in unscale_buttons]
    button_rect = buttons[0].get_rect()
    button_rect.center = position
    return buttons[2],buttons[0],buttons[1], button_rect
Easy1, Easy2, Easy3, Easy_rect = get_buttons("Resources/Buttons/Easy",(WINDOWWIDTH//2, WINDOWHEIGHT//2 - 100), (192,64))
Normal1, Normal2, Normal3, Normal_rect = get_buttons("Resources/Buttons/Normal",(WINDOWWIDTH//2, WINDOWHEIGHT//2 - 25), (192,64))
Hard1, Hard2, Hard3, Hard_rect = get_buttons("Resources/Buttons/Hard",(WINDOWWIDTH//2, WINDOWHEIGHT//2 + 50), (192,64) )
sound1, sound2, sound3, sound_rect = get_buttons("Resources/Buttons/Music",(WINDOWWIDTH//2 - 48, WINDOWHEIGHT//2 + 125), (64,64))
back1, back2, back3, back_rect = get_buttons("Resources/Buttons/Back", (WINDOWWIDTH//2 + 48, WINDOWHEIGHT//2 + 125), (64,64))

# Các nút
buttons = {
    "Board: 12 x 7": {
        "image_on" : Easy1,
        "image_off": Easy2,
        "image_hover": Easy3,
        "rect": Easy_rect,
        "mode": True
        },
    "Board: 14 x 8": {
        "image_on" : Normal1,
        "image_off": Normal2,
        "image_hover": Normal3,
        "rect": Normal_rect,
        "mode" : False
    },
    "Board: 18 x 10": {
        "image_on" : Hard1,
        "image_off" : Hard2,
        "image_hover": Hard3,
        "rect": Hard_rect,
        "mode": False
    },
    "sound_toggle": {
        "image_on" : sound1,
        "image_off": sound2,
        "image_hover": sound3,
        "rect": sound_rect,
        "mode": SOUND_ON
    },
    "back": {
        "image_on": back1,
        "image_off": back2,
        "image_hover": back3, 
        "rect": back_rect,
        "mode": True
    },
}

# Làm mờ background
sigma = 5
startBG = pygame.image.load("Resources/Background/startBG.png")
startBG = pygame.transform.scale(startBG, (WINDOWWIDTH, WINDOWHEIGHT))
arr = pygame.surfarray.array3d(startBG)  # Chuyển thành numpy array
blurred_arr = scipy.ndimage.gaussian_filter(arr, sigma=(sigma, sigma, 0))  # Làm mờ Gaussian
blurred_bg =  pygame.surfarray.make_surface(blurred_arr)

settingboard = pygame.image.load("Resources/others/Board.png")
settingboard = pygame.transform.scale(settingboard,(600,550))
settingboard_rect = settingboard.get_rect()
settingboard_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT//2)
setting = pygame.image.load("Resources/others/setting.png")
setting = pygame.transform.scale(setting,(375,100))
setting_rect = setting.get_rect()
setting_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT//2 - 200)

def draw_background():
    """Vẽ Background bị làm mờ"""
    DISPLAYSURF.blit(blurred_bg, (0,0))

def draw_settings_box():
    """Vẽ bảng cài đặt."""
    DISPLAYSURF.blit(settingboard,settingboard_rect)

def draw_title():
    """Vẽ tiêu đề bảng settings"""
    DISPLAYSURF.blit(setting,setting_rect)

# Các nút tùy chỉnh cài đặt
def draw_buttons():
    """Vẽ các nút."""
    for button in buttons.values():
        if button["mode"]:
            DISPLAYSURF.blit(button["image_on"], button["rect"])
        else:
            DISPLAYSURF.blit(button["image_off"], button["rect"])

# Tùy chỉnh các cài đặt phù hợp       
def handle_click_setting():
    """Tùy chỉnh các cài đặt bằng nhấp chuột"""
    global BOARDWIDTH, BOARDHEIGHT, SOUND_ON, XMARGIN, YMARGIN, BOXSIZE, HEROES_DICT, LEVEL, NUMHEROES_ONBOARD, NUMSAMEHEROES
    if buttons["Board: 12 x 7"]["rect"].collidepoint(pygame.mouse.get_pos()):
        DISPLAYSURF.blit(buttons["Board: 12 x 7"]["image_hover"], buttons["Board: 12 x 7"]["rect"])
    if buttons["Board: 14 x 8"]["rect"].collidepoint(pygame.mouse.get_pos()):
        DISPLAYSURF.blit(buttons["Board: 14 x 8"]["image_hover"], buttons["Board: 14 x 8"]["rect"])
    if buttons["Board: 18 x 10"]["rect"].collidepoint(pygame.mouse.get_pos()):
        DISPLAYSURF.blit(buttons["Board: 18 x 10"]["image_hover"], buttons["Board: 18 x 10"]["rect"])
    if buttons["sound_toggle"]["rect"].collidepoint(pygame.mouse.get_pos()):
        DISPLAYSURF.blit(buttons["sound_toggle"]["image_hover"], buttons["sound_toggle"]["rect"])
    if buttons["back"]["rect"].collidepoint(pygame.mouse.get_pos()):
        DISPLAYSURF.blit(buttons["back"]["image_hover"], buttons["back"]["rect"])
        
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            for key, button in buttons.items():
                if button["rect"].collidepoint((mouse_x, mouse_y)):
                    mouseclick()
                    if key == "Board: 12 x 7":
                        buttons["Board: 12 x 7"]["mode"] = True
                        buttons["Board: 18 x 10"]["mode"] = False
                        buttons["Board: 14 x 8"]["mode"] = False
                        BOARDHEIGHT = 9
                        BOARDWIDTH = 14
                        BOXSIZE = 55
                        NUMHEROES_ONBOARD = 21
                        NUMSAMEHEROES = 4
                        XMARGIN = (WINDOWWIDTH - (BOXSIZE * BOARDWIDTH)) // 2
                        YMARGIN = (WINDOWHEIGHT - (BOXSIZE * BOARDHEIGHT)) // 2
                        HEROES_DICT = Createheroes()
                    elif key == "Board: 14 x 8":
                        buttons["Board: 12 x 7"]["mode"] = False
                        buttons["Board: 18 x 10"]["mode"] = False
                        buttons["Board: 14 x 8"]["mode"] = True
                        BOARDHEIGHT = 10
                        BOARDWIDTH = 16
                        BOXSIZE = 50
                        NUMHEROES_ONBOARD = 28
                        NUMSAMEHEROES = 4
                        XMARGIN = (WINDOWWIDTH - (BOXSIZE * BOARDWIDTH)) // 2
                        YMARGIN = (WINDOWHEIGHT - (BOXSIZE * BOARDHEIGHT)) // 2
                        HEROES_DICT = Createheroes()
                    elif key == "Board: 18 x 10":
                        buttons["Board: 12 x 7"]["mode"] = False
                        buttons["Board: 18 x 10"]["mode"] = True
                        buttons["Board: 14 x 8"]["mode"] = False
                        BOARDHEIGHT = 12
                        BOARDWIDTH = 20
                        BOXSIZE = 44
                        NUMHEROES_ONBOARD = 45
                        NUMSAMEHEROES = 4
                        XMARGIN = (WINDOWWIDTH - (BOXSIZE * BOARDWIDTH)) // 2
                        YMARGIN = (WINDOWHEIGHT - (BOXSIZE * BOARDHEIGHT)) // 2
                        HEROES_DICT = Createheroes()
                    elif key == "sound_toggle":
                        SOUND_ON = not SOUND_ON  # Đảo trạng thái âm thanh
                        buttons["sound_toggle"]["mode"] = SOUND_ON
                        if SOUND_ON:
                            try:
                                pygame.mixer.music.load(listMusicBG[LEVEL - 1])  # Tải nhạc nền
                                pygame.mixer.music.play(-1)  # Phát nhạc nền lặp lại
                            except pygame.error as e:
                                print(f"Error loading music: {e}")
                        else:
                            pygame.mixer.music.stop()  # Dừng nhạc
                    elif key == "back":
                        return "back"
                    
#Hiển thị bảng setting của trò chơi
def showSetting():
    """Hiển thị bảng Settings"""
    while True:
        draw_background()
        draw_settings_box()
        draw_buttons()
        draw_title()
        if handle_click_setting() == "back":
            return
        pygame.display.update()

"""BẢNG XẾP HẠNG"""
def save_leaderboard(leaderboard, filename="User_data/scoreboard.json"):
    """Lưu bảng xếp hạng"""
    with open(filename, "w") as file:
        json.dump(leaderboard, file)

def load_leaderboard(filename="User_data/scoreboard.json"):
    """Load bảng xếp hạng"""
    try:
        with open(filename, "r") as file:
            return json.load(file)  # Trả về danh sách
    except FileNotFoundError:
        return {}
    
# Hiển thị bảng xếp hạng của trò chơi
def showLeaderboard(screen, font, WINDOWWIDTH, WINDOWHEIGHT):
    """Hiển thị bảng xếp hạng"""
    clock = pygame.time.Clock()  # Khởi tạo clock đúng cách
    leaderboard = load_leaderboard()  # Đọc dữ liệu từ file
    sorted_keys = sorted(leaderboard, key = lambda x: leaderboard[x], reverse= True)
    if len(sorted_keys) >= 8:
        sorted_keys =sorted_keys[:8]
    running = True
    sigma = 5
    startBG = pygame.image.load("Resources/Background/startBG.png")
    startBG = pygame.transform.scale(startBG, (WINDOWWIDTH, WINDOWHEIGHT))
    arr = pygame.surfarray.array3d(startBG)  # Chuyển thành numpy array
    blurred_arr = scipy.ndimage.gaussian_filter(arr, sigma=(sigma, sigma, 0))  # Làm mờ Gaussian
    blurred_bg =  pygame.surfarray.make_surface(blurred_arr)
    board = pygame.image.load("Resources/others/leaderboard.png")
    board = pygame.transform.scale_by(board,7)
    board_rect = board.get_rect(center = (WINDOWWIDTH//2, WINDOWHEIGHT//2))
    back = pygame.image.load("Resources/Buttons/Back2/A_Back1.png")
    back_hover = pygame.image.load("Resources/Buttons/Back2/B_Back1.png")
    back = pygame.transform.scale_by(back,2.5)
    back_hover = pygame.transform.scale_by(back_hover, 2.5)
    back_rect = back.get_rect(center = (120,60))
    leaderboard_title = pygame.image.load("Resources/others/leaderboard_title.png")
    leaderboard_title = pygame.transform.scale_by(leaderboard_title, 0.6)
    leaderboard_title_rect = leaderboard_title.get_rect(center = (WINDOWWIDTH//2, 65))
    while running:
        screen.blit(blurred_bg, (0,0))  # Nền xanh đậm
        screen.blit(board,board_rect)
        screen.blit(leaderboard_title, leaderboard_title_rect)

        for i, key in enumerate(sorted_keys):  # Hiển thị top 10
            if len(key) > 8:
                text = text = f"{i+1}.  {key[:8]}... - {leaderboard[key]}"
            else:
                text = f"{i+1}.  {key} - {leaderboard[key]}"
            entry_text =  pygame.font.Font("Resources/Fonts/Jersey15-Regular.ttf", 50).render(text, True, (255, 255, 200))
            screen.blit(entry_text, (WINDOWWIDTH//2 - 200, 150 + i * 50))

        # Vẽ nút "Back"
        screen.blit(back,back_rect)
        if back_rect.collidepoint(pygame.mouse.get_pos()):
            screen.blit(back_hover,back_rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Bấm Enter để quay lại màn hình chính
                    running = False  # Thoát khỏi vòng lặp để quay lại màn hình chính
            elif event.type == pygame.MOUSEBUTTONUP:
                mousex, mousey = event.pos
                if back_rect.collidepoint((mousex, mousey)):  # Kiểm tra click vào nút "RETURN"
                    mouseclick()
                    running = False  # Thoát vòng lặp khi click vào nút "RETURN"

        clock.tick(30)  # Điều chỉnh tốc độ FPS (30 FPS)

    # Sau khi thoát vòng lặp, quay lại màn hình start screen
    return "ReturnToStartScreen"  # Trả về giá trị để quay lại màn hình chính

# Make a dict to store scaled images
def Createheroes():
    """Tạo pokemon dựa trên kích cỡ bảng"""
    LISTHEROES = os.listdir("Resources/Pokemon_icons")
    HEROES_DICT = {}

    for i in range(len(LISTHEROES)):
        hero_image = pygame.image.load("Resources/Pokemon_icons/"+ LISTHEROES[i])
        hero_image = pygame.transform.scale(hero_image, (BOXSIZE, BOXSIZE))
        hero_with_border = pygame.Surface((BOXSIZE, BOXSIZE))
        pygame.draw.rect(hero_with_border, BORDERCOLOR, (0, 0, BOXSIZE, BOXSIZE), 3)
        hero_with_border.blit(hero_image, (0, 0))
        HEROES_DICT[i + 1] = hero_with_border
    return HEROES_DICT

"""Màn hình khi dừng trò chơi"""
def showPauseScreen():
    # Nút Resume
    resume1, resume2, resume3, resume_rect = get_buttons("Resources/Buttons/Resume",(WINDOWWIDTH // 2, WINDOWHEIGHT // 2 - 40), (192,64))
    
    # Nút Exit
    exit1, exit2, exit3, exit_rect = get_buttons("Resources/Buttons/Exit",(WINDOWWIDTH // 2, WINDOWHEIGHT // 2 + 40) ,(192,64) )

    Board = pygame.image.load("Resources/others/Board.png")
    Board = pygame.transform.scale(Board, (360,330))
    Board_rect = Board.get_rect()
    Board_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT // 2)
    while True:
        DISPLAYSURF.fill(BLACK)  # Màn hình tạm dừng
        
        DISPLAYSURF.blit(Board,Board_rect)

        # Vẽ nút Resume
        DISPLAYSURF.blit(resume2, resume_rect)

        # Vẽ nút Exit
        DISPLAYSURF.blit(exit2, exit_rect) 

        if resume_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(resume3, resume_rect)
        if exit_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(exit3, exit_rect)
        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                mousex, mousey = event.pos
                if resume_rect.collidepoint((mousex, mousey)):
                    mouseclick()
                    return  # Quay lại game
                elif exit_rect.collidepoint((mousex, mousey)):
                    mouseclick()
                    return "exit"  # Quay về màn hình chính
        pygame.display.update()

"""Lưu lại màn chơi"""
def saveGame(mainBoard):
    USER["Board"] = mainBoard
    USER["Score"] = SCORE
    USER["Level"] = LEVEL
    with open(ACCOUNTS_FILE, "r") as file:
        data = json.load(file)
    data[USER_NAME] = USER
    with open(ACCOUNTS_FILE, "w") as file:
        json.dump(data, file)

def handleHintMatch(mainBoard, boxy1, boxx1, boxy2, boxx2, hint):
    global TIMEBONUS
    mainBoard[boxy1][boxx1] = 0
    mainBoard[boxy2][boxx2] = 0
    TIMEBONUS += 1
    alterBoardWithLevel(mainBoard, boxy1, boxx1, boxy2, boxx2, LEVEL)
    if isGameComplete(mainBoard):
        drawBoard(mainBoard)
        pygame.display.update()
        return
    if not bfs(mainBoard, hint[0][0], hint[0][1], hint[1][0], hint[1][1]):
        hint[:] = getHint(mainBoard)

def handleSelection(mainBoard, firstSelection, secondSelection, clickedBoxes, hint):
    if bfs(mainBoard, firstSelection[1], firstSelection[0], secondSelection[1], secondSelection[0]):
        mainBoard[firstSelection[1]][firstSelection[0]] = 0
        mainBoard[secondSelection[1]][secondSelection[0]] = 0
        drawPath(mainBoard, bfs(mainBoard, firstSelection[1], firstSelection[0], secondSelection[1], secondSelection[0]))
        # Gọi hàm alterBoardWithLevel
        alterBoardWithLevel(mainBoard, firstSelection[1], firstSelection[0], secondSelection[1], secondSelection[0], LEVEL)
        hint = getHint(mainBoard)

def isGameComplete(board):
    """
    Kiểm tra xem bảng đã hoàn thành hay chưa.
    Trả về True nếu tất cả các ô trên bảng đều là 0.
    """
    for row in board:
        if any(cell != 0 for cell in row):  # Nếu còn ô nào khác 0
            return False
    return True  # Tất cả các ô đều là 0

def drawSideButton(img, img_hover, rect, is_disabled, flash, flash_start, label_surf, label_rect):
    """Vẽ một nút bên cạnh khung game (dùng chung cho Shuffle và Restart)"""
    mouse_pos = pygame.mouse.get_pos()
    is_hovering = (not is_disabled) and rect.collidepoint(mouse_pos)
    is_flashing = flash and (time.time() - flash_start < 0.35)

    # --- Panel nền bo góc ---
    panel_rect = rect.inflate(20, 16)
    panel_surf = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    if is_disabled:
        panel_surf.fill((40, 40, 40, 100))
        pygame.draw.rect(panel_surf, (90, 90, 90, 120), panel_surf.get_rect(), 2, border_radius=12)
    elif is_flashing:
        panel_surf.fill((255, 210, 0, 100))
        pygame.draw.rect(panel_surf, (255, 230, 0, 200), panel_surf.get_rect(), 2, border_radius=12)
    elif is_hovering:
        panel_surf.fill((50, 150, 220, 80))
        pygame.draw.rect(panel_surf, (80, 200, 255, 180), panel_surf.get_rect(), 2, border_radius=12)
    else:
        panel_surf.fill((15, 20, 45, 120))
        pygame.draw.rect(panel_surf, (80, 110, 180, 140), panel_surf.get_rect(), 2, border_radius=12)
    DISPLAYSURF.blit(panel_surf, panel_rect.topleft)

    # --- Ảnh nút (mờ nếu disabled) ---
    draw_img = img_hover if (is_hovering or is_flashing) else img
    if is_disabled:
        disabled_img = draw_img.copy()
        disabled_img.set_alpha(80)
        DISPLAYSURF.blit(disabled_img, rect)
    else:
        DISPLAYSURF.blit(draw_img, rect)

    # --- Label ---
    DISPLAYSURF.blit(label_surf, label_rect)

def runGame():
    if USER["Board"] == [[0] * BOARDWIDTH] * BOARDHEIGHT:
        USER["Board"] = getRandomizedBoard()
    mainBoard = USER["Board"]
    pygame.draw.rect(DISPLAYSURF, BORDERCOLOR,
                     (XMARGIN - 3, YMARGIN - 3, BOARDWIDTH * BOXSIZE + 6, BOARDHEIGHT * BOXSIZE + 6), 6)
    clickedBoxes = []
    firstSelection = None
    mousex, mousey = 0, 0
    hint = None  # Gợi ý mặc định là None
    hint_visible = False  # Biến cờ để theo dõi trạng thái hiển thị gợi ý
    hint_display_start = 0  # Thời điểm bắt đầu hiển thị gợi ý
    hint_display_time = 1  # Gợi ý sẽ hiển thị trong 3 giây

    global GAMETIME, LIVES, TIMEBONUS, STARTTIME, LEVEL, SCORE, SOUND_ON
    STARTTIME = time.time()
    TIMEBONUS = 0
    randomBG = BG

    # --- Nút Shuffle ---
    SHUFFLE_MAX = 9
    shuffle_count = 0
    btn_size = (72, 72)
    side_font = pygame.font.Font(font_path, 20)
    board_right = XMARGIN + BOARDWIDTH * BOXSIZE
    btn_centerx = board_right + (WINDOWWIDTH - board_right) // 2

    # Load và xử lý ảnh Shuffle (loại bỏ nền đen)
    shuffle_img_orig = pygame.image.load('Resources/Buttons/Shuffle/Shuffle.png').convert()
    shuffle_img_orig.set_colorkey((0, 0, 0))  # Bỏ nền đen
    shuffle_img = pygame.transform.smoothscale(shuffle_img_orig, btn_size)
    shuffle_img_hover = shuffle_img.copy()
    hover_ov = pygame.Surface(btn_size, pygame.SRCALPHA)
    hover_ov.fill((255, 255, 255, 55))
    shuffle_img_hover.blit(hover_ov, (0, 0))
    shuffle_rect = shuffle_img.get_rect(centerx=btn_centerx, centery=WINDOWHEIGHT // 2 - 60)
    shuffle_flash = False
    shuffle_flash_start = 0

    # Load và xử lý ảnh Restart
    restart_img_orig = pygame.image.load('Resources/Buttons/Restart/restart.png').convert_alpha()
    restart_img = pygame.transform.smoothscale(restart_img_orig, btn_size)
    restart_img_hover = restart_img.copy()
    restart_img_hover.blit(hover_ov, (0, 0))
    restart_rect = restart_img.get_rect(centerx=btn_centerx, top=shuffle_rect.bottom + 28)
    restart_flash = False
    restart_flash_start = 0

    # Load và xử lý ảnh Home
    home_img_orig = pygame.image.load('Resources/Buttons/Home/home-button.png').convert_alpha()
    home_img = pygame.transform.smoothscale(home_img_orig, btn_size)
    home_img_hover = home_img.copy()
    home_img_hover.blit(hover_ov, (0, 0))
    home_rect = home_img.get_rect(centerx=btn_centerx, top=restart_rect.bottom + 28)
    home_flash = False
    home_flash_start = 0

    def make_shuffle_label():
        remaining = SHUFFLE_MAX - shuffle_count
        color = (255, 80, 80) if remaining <= 3 else (255, 230, 80)
        lbl = side_font.render(f"SHUFFLE ({remaining})", True, color)
        return lbl, lbl.get_rect(centerx=shuffle_rect.centerx, top=shuffle_rect.bottom + 5)

    shuffle_label, shuffle_label_rect = make_shuffle_label()
    restart_label = side_font.render("RESTART", True, (180, 230, 255))
    restart_label_rect = restart_label.get_rect(centerx=restart_rect.centerx, top=restart_rect.bottom + 5)
    home_label = side_font.render("HOME", True, (160, 220, 160))
    home_label_rect = home_label.get_rect(centerx=home_rect.centerx, top=home_rect.bottom + 5)

    # Phát nhạc nền
    pygame.mixer.music.load(listMusicBG[LEVEL - 1])
    if SOUND_ON:
        try:
            pygame.mixer.music.play(-1)
        except pygame.error as e:
            print(f"Error playing music: {e}")
    else:
        pygame.mixer.music.stop()

    ai_selected_pair = []
    ai_click_timer = time.time()
    ai_click_step = 0

    while True:
        DISPLAYSURF.blit(randomBG, (0, 0))
        drawBoard(mainBoard)
        drawTimeBar()
        drawLives()
        display_score(SCORE)

        # Vẽ nút Shuffle, Restart và Home
        shuffle_label, shuffle_label_rect = make_shuffle_label()
        drawSideButton(shuffle_img, shuffle_img_hover, shuffle_rect,
                       shuffle_count >= SHUFFLE_MAX, shuffle_flash, shuffle_flash_start,
                       shuffle_label, shuffle_label_rect)
        drawSideButton(restart_img, restart_img_hover, restart_rect,
                       False, restart_flash, restart_flash_start,
                       restart_label, restart_label_rect)
        drawSideButton(home_img, home_img_hover, home_rect,
                       False, home_flash, home_flash_start,
                       home_label, home_label_rect)

        # Kiểm tra nếu hoàn thành bảng
        if isGameComplete(mainBoard):
            pygame.time.wait(1000)  # Tạm dừng để hiển thị
            return True  # Hoàn thành màn chơi

        # Kiểm tra thời gian hết game
        if time.time() - STARTTIME > GAMETIME + TIMEBONUS:
            LEVEL = LEVELMAX + 1 # Hiển thị màn hình kết thúc
            showGameOverScreen()
            return

        # Ẩn gợi ý sau khoảng thời gian hiển thị
        if hint_visible and time.time() - hint_display_start > hint_display_time:
            hint_visible = False  # Đặt lại cờ, gợi ý sẽ không hiển thị nữa

        if hint_visible:  # Chỉ vẽ gợi ý khi đang ở trạng thái hiển thị
            drawHint(hint)

        mouseClicked = False
        
        if AUTOPILOT:
            if not ai_selected_pair:
                valid_actions = get_valid_actions(mainBoard, bfs)
                if valid_actions:
                    s_dict = extract_state(mainBoard)
                    s_tensor = state_to_torch(s_dict, device="cpu")["board_onehot"].unsqueeze(0)
                    with torch.no_grad():
                        q_values = ai_model(s_tensor)
                    best_action = qvalues_to_action(q_values[0], valid_actions, BOARDWIDTH, BOARDHEIGHT)
                    if best_action is None:
                        best_action = random.choice(valid_actions)
                    r1, c1, r2, c2 = best_action
                    ai_selected_pair = [(c1, r1), (c2, r2)]
                    ai_click_step = 0
                    ai_click_timer = time.time()
            
            if ai_selected_pair:
                if ai_click_step == 0 and time.time() - ai_click_timer > 0.3:
                    mousex, mousey = leftTopCoordsOfBox(ai_selected_pair[0][0], ai_selected_pair[0][1])
                    mousex += BOXSIZE // 2
                    mousey += BOXSIZE // 2
                    mouseClicked = True
                    ai_click_step = 1
                    ai_click_timer = time.time()
                elif ai_click_step == 1 and time.time() - ai_click_timer > 0.3:
                    mousex, mousey = leftTopCoordsOfBox(ai_selected_pair[1][0], ai_selected_pair[1][1])
                    mousex += BOXSIZE // 2
                    mousey += BOXSIZE // 2
                    mouseClicked = True
                    ai_selected_pair = []
                    ai_click_timer = time.time()

        hint = getHint(mainBoard)
        if not hint:
            resetBoard(mainBoard)
        for event in pygame.event.get():
            if event.type == QUIT:
                saveGame(mainBoard)
                pygame.quit()
                sys.exit()
            # event.type == MOUSEBUTTONDOWN
            elif event.type == MOUSEBUTTONDOWN:
                mousex, mousey = event.pos
                mouseClicked = True
                # Kiểm tra click vào nút Shuffle
                if shuffle_rect.collidepoint(mousex, mousey) and shuffle_count < SHUFFLE_MAX:
                    mouseclick()
                    resetBoard(mainBoard)
                    shuffle_count += 1
                    firstSelection = None
                    clickedBoxes.clear()
                    shuffle_flash = True
                    shuffle_flash_start = time.time()
                    mousex, mousey = 0, 0
                    mouseClicked = False
                # Kiểm tra click vào nút Restart
                elif restart_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    mainBoard = getRandomizedBoard()
                    USER["Board"] = mainBoard
                    shuffle_count = 0
                    firstSelection = None
                    clickedBoxes.clear()
                    STARTTIME = time.time()
                    TIMEBONUS = 0
                    restart_flash = True
                    restart_flash_start = time.time()
                    mousex, mousey = 0, 0
                    mouseClicked = False
                # Kiểm tra click vào nút Home
                elif home_rect.collidepoint(mousex, mousey):
                    mouseclick()
                    saveGame(mainBoard)
                    home_flash = True
                    home_flash_start = time.time()
                    pygame.time.wait(250)
                    return "home"  # Quảy về menu chính
            elif event.type == KEYUP:
                if event.key == K_ESCAPE:
                    result = showPauseScreen()
                    if result == "exit":
                        saveGame(mainBoard)
                        return False  # Chỉ hiển thị gợi ý khi bấm phím N
                hint = getHint(mainBoard)  # Lấy gợi ý
                if event.key == K_n:  # Nếu có gợi ý hợp lệ
                    SCORE = max(SCORE - 1, 0)
                    hint_visible = True  # Bật cờ hiển thị gợi ý
                    hint_display_start = time.time()  # Đặt thời gian bắt đầu hiển thị
                if event.key == K_r:
                    resetBoard(mainBoard)

        # Xử lý lựa chọn trên bảng
        boxx, boxy = getBoxAtPixel(mousex, mousey)
        if firstSelection:
                drawHighlightBox(mainBoard, boxx, boxy, color=YELLOW)
        if boxx is not None and boxy is not None and mainBoard[boxy][boxx] != 0 and mouseClicked:
            clickedBoxes.append((boxx, boxy))
            drawClickedBox(mainBoard, clickedBoxes)
            if firstSelection is None:  # Nếu đây là lựa chọn đầu tiên
                firstSelection = (boxx, boxy)
                clickSound.play()
            else:  # Đã có lựa chọn đầu tiên, kiểm tra cặp ô
                if bfs(mainBoard, firstSelection[1], firstSelection[0], boxy, boxx):  # Nếu hợp lệ
                    handleSelection(mainBoard, firstSelection, (boxx, boxy), clickedBoxes, hint)
                    firstSelection = None
                    pygame.mixer.Sound.play(getPointSound)
                    clickedBoxes.clear()  # Reset danh sách sau khi hoàn thành
                else:  # Không hợp lệ
                    drawHighlightBox(mainBoard, firstSelection[0], firstSelection[1], color=RED)
                    drawHighlightBox(mainBoard, boxx, boxy, color=RED)
                    pygame.mixer.Sound.play(WrongSound)
                    pygame.display.update()
                    pygame.time.wait(300)
                    clickedBoxes.clear()
                    firstSelection = None

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def getRandomizedBoard():
    """Tạo bảng ngẫu nhiên - đảm bảo mỗi loại xuất hiện đúng NUMSAMEHEROES lần"""
    inner_cells = (BOARDHEIGHT - 2) * (BOARDWIDTH - 2)
    # Số loại pokemon cần (tính từ kích thước bảng thực tế)
    num_types = inner_cells // NUMSAMEHEROES
    # Nếu inner_cells không chia hết, cắt bớt để đảm bảo cặp đôi
    usable_cells = num_types * NUMSAMEHEROES

    all_pokemons = list(range(1, len(Createheroes()) + 1))
    random.shuffle(all_pokemons)

    # Nếu không đủ loại pokemon, lầy lại từ đầu (tránh lỗi với mode Hard)
    full_cycles = (num_types // len(all_pokemons)) + 1
    pool = (all_pokemons * full_cycles)[:num_types]

    list_pokemons = pool * NUMSAMEHEROES  # Mỗi loại xuất hiện đúng NUMSAMEHEROES lần
    random.shuffle(list_pokemons)

    board = [[0 for _ in range(BOARDWIDTH)] for _ in range(BOARDHEIGHT)]
    k = 0
    for i in range(1, BOARDHEIGHT - 1):
        for j in range(1, BOARDWIDTH - 1):
            if k < usable_cells:
                board[i][j] = list_pokemons[k]
                k += 1
            # Các ô thừa (không chia hết) giữ nguyên là 0
    return board

def leftTopCoordsOfBox(boxx, boxy):
    left = boxx * BOXSIZE + XMARGIN
    top = boxy * BOXSIZE + YMARGIN
    return left, top

def getBoxAtPixel(x, y):
    if x <= XMARGIN or x >= WINDOWWIDTH - XMARGIN or y <= YMARGIN or y >= WINDOWHEIGHT - YMARGIN:
        return None, None
    return (x - XMARGIN) // BOXSIZE, (y - YMARGIN) // BOXSIZE

def drawBoard(board):
    a = Createheroes()

    # --- Vẽ lưới (grid) cho vùng ô bên trong ---
    inner_left  = XMARGIN + BOXSIZE
    inner_top   = YMARGIN + BOXSIZE
    inner_w     = (BOARDWIDTH  - 2) * BOXSIZE
    inner_h     = (BOARDHEIGHT - 2) * BOXSIZE
    cols = BOARDWIDTH  - 2
    rows = BOARDHEIGHT - 2

    # --- Bước 1: Vẽ nền tối nhẹ ---
    bg_surf = pygame.Surface((inner_w, inner_h), pygame.SRCALPHA)
    bg_surf.fill((10, 15, 40, 55))
    DISPLAYSURF.blit(bg_surf, (inner_left, inner_top))

    # --- Bước 2: Vẽ icon pokemon lên trên nền ---
    for boxx in range(BOARDWIDTH):
        for boxy in range(BOARDHEIGHT):
            if board[boxy][boxx] != 0:
                left, top = leftTopCoordsOfBox(boxx, boxy)
                boxRect = pygame.Rect(left, top, BOXSIZE, BOXSIZE)
                DISPLAYSURF.blit(a[board[boxy][boxx]], boxRect)

    # --- Bước 3: Vẽ đường lưới LÊN TRÊN icon (luôn hiển thị) ---
    line_surf = pygame.Surface((inner_w, inner_h), pygame.SRCALPHA)
    line_color = (80, 110, 200, 160)
    for c in range(cols + 1):
        x = c * BOXSIZE
        pygame.draw.line(line_surf, line_color, (x, 0), (x, inner_h), 1)
    for r in range(rows + 1):
        y = r * BOXSIZE
        pygame.draw.line(line_surf, line_color, (0, y), (inner_w, y), 1)
    DISPLAYSURF.blit(line_surf, (inner_left, inner_top))

def drawHighlightBox(board, boxx, boxy, color=HIGHLIGHTCOLOR):
    left, top = leftTopCoordsOfBox(boxx, boxy)
    pygame.draw.rect(DISPLAYSURF, color, (left - 2, top - 2, BOXSIZE + 4, BOXSIZE + 4), 2)

def drawClickedBox(board, clickedBoxes):
    a = Createheroes()
    for boxx, boxy in clickedBoxes:
        left, top = leftTopCoordsOfBox(boxx, boxy)
        boxRect = pygame.Rect(left, top, BOXSIZE, BOXSIZE)
        image = a[board[boxy][boxx]].copy()

        # Darken the clicked image
        image.fill((60, 60, 60), special_flags=pygame.BLEND_RGB_SUB)
        DISPLAYSURF.blit(image, boxRect)

def bfs(board, boxy1, boxx1, boxy2, boxx2):
    def backtrace(parent, boxy1, boxx1, boxy2, boxx2):
        start = (boxy1, boxx1, 0, 'no_direction')
        end = 0
        for node in parent:
            if node[:2] == (boxy2, boxx2):
                end = node

        path = [end]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()

        for i in range(len(path)):
            path[i] = path[i][:2]
        return path

    if board[boxy1][boxx1] != board[boxy2][boxx2]:
        return []

    n = len(board)
    m = len(board[0])

    import collections
    q = collections.deque()
    q.append((boxy1, boxx1, 0, 'no_direction'))
    visited = set()
    visited.add((boxy1, boxx1, 0, 'no_direction'))
    parent = {}

    while len(q) > 0:
        r, c, num_turns, direction = q.popleft()
        if (r, c) != (boxy1, boxx1) and (r, c) == (boxy2, boxx2):
            return backtrace(parent, boxy1, boxx1, boxy2, boxx2)

        dict_directions = {(r + 1, c): 'down', (r - 1, c): 'up', (r, c - 1): 'left',
                           (r, c + 1): 'right'}
        for neiborX, neiborY in dict_directions:
            next_direction = dict_directions[(neiborX, neiborY)]
            if 0 <= neiborX <= n - 1 and 0 <= neiborY <= m - 1 and (
                    board[neiborX][neiborY] == 0 or (neiborX, neiborY) == (boxy2, boxx2)):
                if direction == 'no_direction':
                    q.append((neiborX, neiborY, num_turns, next_direction))
                    visited.add((neiborX, neiborY, num_turns, next_direction))
                    parent[(neiborX, neiborY, num_turns, next_direction)] = (
                    r, c, num_turns, direction)
                elif direction == next_direction and (
                        neiborX, neiborY, num_turns, next_direction) not in visited:
                    q.append((neiborX, neiborY, num_turns, next_direction))
                    visited.add((neiborX, neiborY, num_turns, next_direction))
                    parent[(neiborX, neiborY, num_turns, next_direction)] = (
                    r, c, num_turns, direction)
                elif direction != next_direction and num_turns < 2 and (
                        neiborX, neiborY, num_turns + 1, next_direction) not in visited:
                    q.append((neiborX, neiborY, num_turns + 1, next_direction))
                    visited.add((neiborX, neiborY, num_turns + 1, next_direction))
                    parent[
                        (neiborX, neiborY, num_turns + 1, next_direction)] = (
                    r, c, num_turns, direction)
    return []

def getCenterPos(pos): # pos is coordinate of a box in mainBoard
    left, top = leftTopCoordsOfBox(pos[1], pos[0])
    return tuple([left + BOXSIZE // 2, top + BOXSIZE // 2])

def drawPath(board, path):
    global SCORE
    for i in range(len(path) - 1):
        startPos = getCenterPos(path[i])
        endPos = getCenterPos(path[i + 1])
        pygame.draw.line(DISPLAYSURF, RED, startPos, endPos, 4)
    SCORE += 1
    pygame.display.update()
    pygame.time.wait(300)

def drawTimeBar():
    progress = 1 - ((time.time() - STARTTIME - TIMEBONUS) / GAMETIME)

    pygame.draw.rect(DISPLAYSURF, borderColor, (barPos, barSize), 1)
    innerPos = (barPos[0] + 2, barPos[1] + 2)
    innerSize = ((barSize[0] - 4) * progress, barSize[1] - 4)
    pygame.draw.rect(DISPLAYSURF, barColor, (innerPos, innerSize))

def showGameOverScreen():
    try:
        # Đọc dữ liệu từ tệp, nếu không có thì tạo tệp
        with open("User_data/scoreboard.json", "r") as leaderboard:
            data = json.load(leaderboard)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}  # Nếu không tồn tại, bắt đầu với danh sách trống

    # Xử lý điểm số người chơi
    if USER_NAME in data:
        data[USER_NAME] = max(SCORE, data[USER_NAME])
    else:
        data[USER_NAME] = SCORE

    # Ghi lại tệp
    with open("User_data/scoreboard.json", "w") as leaderboard:
        json.dump(data, leaderboard, indent=4)

    # Hiển thị màn hình kết thúc
    exit = pygame.image.load("Resources/Buttons/Exit/A_Exit1.png")
    exit_hover = pygame.image.load("Resources/Buttons/Exit/C_Exit1.png")
    exit = pygame.transform.scale2x(exit)
    exit_hover = pygame.transform.scale2x(exit_hover)
    exit_rect = exit.get_rect()
    exit_rect.center = (WINDOWWIDTH//2, WINDOWHEIGHT - 50)

    while True:
        DISPLAYSURF.blit(exit, exit_rect)
        if exit_rect.collidepoint(pygame.mouse.get_pos()):
            DISPLAYSURF.blit(exit_hover, exit_rect)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONUP:
                mousex, mousey = event.pos
                if exit_rect.collidepoint((mousex, mousey)):
                    mouseclick()
                    return
        pygame.display.update()

def getHint(board):
    boxPokesLocated = collections.defaultdict(list)
    for boxy in range(BOARDHEIGHT):
        for boxx in range(BOARDWIDTH):
            if board[boxy][boxx] != 0:
                boxPokesLocated[board[boxy][boxx]].append((boxy, boxx))
    for boxy in range(BOARDHEIGHT):
        for boxx in range(BOARDWIDTH):
            if board[boxy][boxx] != 0:
                for otherBox in boxPokesLocated[board[boxy][boxx]]:
                    if otherBox != (boxy, boxx) and bfs(board, boxy, boxx, otherBox[0], otherBox[1]):
                        return [(boxy, boxx), otherBox]  # Trả về cặp gợi ý
    return []  # Không có gợi ý nào, trả về danh sách rỗng
def drawHint(hint):
    for boxy, boxx in hint:
        left, top = leftTopCoordsOfBox(boxx, boxy)
        pygame.draw.rect(DISPLAYSURF, GREEN, (left, top, BOXSIZE, BOXSIZE), 4)  # Viền dày hơn để rõ

def resetBoard(board):
    pokesOnBoard = []
    for boxy in range(BOARDHEIGHT):
        for boxx in range(BOARDWIDTH):
            if board[boxy][boxx] != 0:
                pokesOnBoard.append(board[boxy][boxx])
    referencedList = pokesOnBoard[:]
    while referencedList == pokesOnBoard:
        random.shuffle(pokesOnBoard)

    i = 0
    for boxy in range(BOARDHEIGHT):
        for boxx in range(BOARDWIDTH):
            if board[boxy][boxx] != 0:
                board[boxy][boxx] = pokesOnBoard[i]
                i += 1
    return board

def isGameComplete(board):
    for boxy in range(BOARDHEIGHT):
        for boxx in range(BOARDWIDTH):
            if board[boxy][boxx] != 0:
                return False
    return True

def alterBoardWithLevel(board, boxy1, boxx1, boxy2, boxx2, level):

    # Level 2: All the pokemons move up to the top boundary
    if level == 2:
        for boxx in (boxx1, boxx2):
            # rearrange pokes into a current list
            cur_list = [0]
            for i in range(BOARDHEIGHT):
                if board[i][boxx] != 0:
                    cur_list.append(board[i][boxx])
            while len(cur_list) < BOARDHEIGHT:
                cur_list.append(0)

            # add the list into the board
            j = 0
            for num in cur_list:
                board[j][boxx] = num
                j += 1

    # Level 3: All the pokemons move down to the bottom boundary
    if level == 3:
        for boxx in (boxx1, boxx2):
            # rearrange pokes into a current list
            cur_list = []
            for i in range(BOARDHEIGHT):
                if board[i][boxx] != 0:
                    cur_list.append(board[i][boxx])
            cur_list.append(0)
            cur_list = [0] * (BOARDHEIGHT - len(cur_list)) + cur_list

            # add the list into the board
            j = 0
            for num in cur_list:
                board[j][boxx] = num
                j += 1

    # Level 4: All the pokemons move left to the left boundary
    if level == 4:
        for boxy in (boxy1, boxy2):
            # rearrange pokes into a current list
            cur_list = [0]
            for i in range(BOARDWIDTH):
                if board[boxy][i] != 0:
                    cur_list.append(board[boxy][i])
            while len(cur_list) < BOARDWIDTH:
                cur_list.append(0)

            # add the list into the board
            j = 0
            for num in cur_list:
                board[boxy][j] = num
                j += 1

    # Level 5: All the pokemons move right to the right boundary
    if level == 5:
        for boxy in (boxy1, boxy2):
            # rearrange pokes into a current list
            cur_list = []
            for i in range(BOARDWIDTH):
                if board[boxy][i] != 0:
                    cur_list.append(board[boxy][i])
            cur_list.append(0)
            cur_list = [0] * (BOARDWIDTH - len(cur_list)) + cur_list

            # add the list into the board
            j = 0
            for num in cur_list:
                board[boxy][j] = num
                j += 1

    return board

def drawLives():
    aegisRect = pygame.Rect(10, 10, BOXSIZE, BOXSIZE)
    DISPLAYSURF.blit(aegis, aegisRect)
    livesSurf = LIVESFONT.render(str(LIVES), True, WHITE)
    livesRect = livesSurf.get_rect()
    livesRect.topleft = (65, 0)
    DISPLAYSURF.blit(livesSurf, livesRect)

def change_size():
    global XMARGIN, YMARGIN, BOXSIZE, barPos
    if BOARDWIDTH == 14:
        BOXSIZE = 66
    elif BOARDWIDTH == 20:
        BOXSIZE = 46
    XMARGIN = (WINDOWWIDTH - (BOXSIZE * BOARDWIDTH)) // 2
    YMARGIN = (WINDOWHEIGHT - (BOXSIZE * BOARDHEIGHT)) // 2
    # Cập nhật vị trí thanh thời gian theo YMARGIN mới
    barPos = (WINDOWWIDTH // 2 - TIMEBAR_LENGTH // 2, max(YMARGIN // 2 - TIMEBAR_WIDTH // 2, 4))

def display_score(score = 0):
    ScoreFont = pygame.font.Font('freesansbold.ttf', 30)
    ScoreSurf = ScoreFont.render(f'Score: {score}', True, YELLOW)
    ScoreRect = ScoreSurf.get_rect()
    ScoreRect.center = (9 * WINDOWWIDTH // 10, WINDOWHEIGHT // 10)
    DISPLAYSURF.blit(ScoreSurf, ScoreRect)
    pygame.draw.rect(DISPLAYSURF, YELLOW, ScoreRect, -1)
    # Không gọi pygame.display.update() ở đây để tránh nhấp nháy


def main():
    """Hàm chính của trò chơi"""
    global FPSCLOCK, DISPLAYSURF, BASICFONT, LIVESFONT, LEVEL, font, title_font, USER, SCORE, BOARDWIDTH, BOARDHEIGHT
    
    # Khởi tạo Pygame
    pygame.init()
    pygame.mixer.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption('Pikachu')

    # Khởi tạo font
    font = pygame.font.Font(pygame.font.match_font('arial'), 32)
    title_font = pygame.font.Font(pygame.font.match_font('arial'), 50)
    BASICFONT = pygame.font.SysFont('pixel', 40)
    LIVESFONT = pygame.font.SysFont('comicsansms', 45)
    
    # Hiển thị màn hình xác thực (đăng nhập/đăng ký)
    if not AUTOPILOT:
        while showMainAuthScreen():
            continue

    # Vòng lặp chính của trò chơi
    while True:
        # Hiển thị màn hình bắt đầu và nhận lựa chọn từ người chơi
        if AUTOPILOT:
            choice = "Newgame"
        else:
            choice = showStartScreen()
        
        # Biến cờ để kiểm tra trạng thái chơi
        game_running = True

        # Xử lý các lựa chọn từ màn hình bắt đầu
        if choice == "Newgame" or choice == "Continue":
            if choice == "Newgame":
                LEVEL = 1
                SCORE = 0
                USER["Board"] = getRandomizedBoard()
            else:
                LEVEL = USER["Level"]
                SCORE = USER["Score"]
                BOARDWIDTH = len(USER["Board"][0])
                BOARDHEIGHT = len(USER["Board"])
            
            # Cập nhật kích thước và tài nguyên
            change_size()
            random.shuffle(listMusicBG)
            
            # Vòng lặp cấp độ
            while LEVEL <= LEVELMAX and game_running:
                result = runGame()
                if result == "home":
                    # Quay về màn hình chính (start screen)
                    pygame.mixer.music.stop()
                    game_running = False
                    break
                elif not result:
                    game_running = False
                    break
                LEVEL += 1
                pygame.time.wait(1000)  # Tạm dừng giữa các cấp độ
            
            # Hiển thị màn hình kết thúc nếu hoàn thành trò chơi
            if game_running:
                showGameOverScreen()
        elif choice == "Setting":
            # Hiển thị cài đặt
            showSetting()


if __name__ == '__main__':
    while True:
        main()  # Chạy lại hàm main khi người chơi nhấn "Play Again"