# -*- coding: utf-8 -*-
with open('pikachu.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the grid_surf block inside drawBoard and restructure it
# We look for the line that starts the old grid drawing and rewrite that section

new_drawboard_inner = '''\
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
'''

# Find start line: "    grid_surf = pygame.Surface((inner_w, inner_h), pygame.SRCALPHA)"
# Find end line: "                DISPLAYSURF.blit(a[board[boxy][boxx]], boxRect)"
# These are inside def drawBoard

start_idx = None
end_idx = None
in_drawboard = False

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == 'def drawBoard(board):':
        in_drawboard = True
    if in_drawboard:
        if 'grid_surf = pygame.Surface((inner_w, inner_h), pygame.SRCALPHA)' in line and start_idx is None:
            start_idx = i
        if start_idx is not None and 'DISPLAYSURF.blit(a[board[boxy][boxx]], boxRect)' in line:
            end_idx = i
            break

print(f"Found block at lines {start_idx+1} to {end_idx+1}")

if start_idx is not None and end_idx is not None:
    # Detect line ending from original file
    eol = '\r\n' if '\r\n' in lines[start_idx] else '\n'
    new_lines_raw = new_drawboard_inner.replace('\n', eol)
    
    new_lines = lines[:start_idx] + [new_lines_raw] + lines[end_idx+1:]
    with open('pikachu.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("SUCCESS: Grid draw order fixed!")
else:
    print("ERROR: Could not find target block")
    if start_idx:
        print(f"start_idx={start_idx}, end_idx={end_idx}")
