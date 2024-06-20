import pygame, random, time

screen_size = 600
board_size = 400
cell_size = 100

bg = pygame.image.load("bg.png")

pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

board = [[None for _ in range(4)] for _ in range(4)]

colors = {
    2: (255, 161, 125),
    4: (255, 205, 125),
    8: (255, 184, 125),
    16: (255, 138, 125),
    32: (255, 220, 125),
    64: (255, 200, 125),
    128: (176, 100, 70),
    256: (191, 145, 71),
    512: (128, 53, 45),
    1024: (217, 178, 72),
    2048: (255, 198, 125),
    4096: (255, 156, 125),
}

def move_board_up(board):
    old_board = [row.copy() for row in board]
    for row in range(1, len(board)):
        if not any(board[row]): continue
        for col in range(len(board)):
            if not board[row][col]: continue
            above = [r[col] for r in board[:row]]
            if not any(above):
                board[0][col] = board[row][col]
                board[row][col] = None
                continue
            closest = None
            for i, cell in list(enumerate(above))[::-1]:
                if cell:
                    closest = i
                    break
            if board[closest][col] == board[row][col]:
                board[closest][col] *= 2
                board[row][col] = None
            elif not board[closest+1][col]:
                board[closest+1][col] = board[row][col]
                board[row][col] = None
    return board, old_board != board

def move_board_down(board):
    old_board = [row.copy() for row in board]
    for row in range(len(board)-2, -1, -1):
        if not any(board[row]): continue
        for col in range(len(board)):
            if not board[row][col]: continue
            below = [r[col] for r in board[row+1:]]
            if not any(below):
                board[-1][col] = board[row][col]
                board[row][col] = None
                continue
            closest = None
            for i, cell in list(enumerate(below))[::-1]:
                if cell:
                    closest = i
                    break
            if board[row+closest+1][col] == board[row][col]:
                board[row+closest+1][col] *= 2
                board[row][col] = None
            elif not board[row+closest][col]:
                board[row+closest][col] = board[row][col]
                board[row][col] = None
    return board, old_board != board

def move_board_left(board):
    old_board = [row.copy() for row in board]
    for col in range(1, len(board)):
        if not any([r[col] for r in board]): continue
        for row in range(len(board)):
            if not board[row][col]: continue
            left = board[row][:col]
            if not any(left):
                board[row][0] = board[row][col]
                board[row][col] = None
                continue
            closest = None
            for i, cell in list(enumerate(left))[::-1]:
                if cell:
                    closest = i
                    break
            if board[row][closest] == board[row][col]:
                board[row][closest] *= 2
                board[row][col] = None
            elif not board[row][closest+1]:
                board[row][closest+1] = board[row][col]
                board[row][col] = None
    return board, old_board != board

def move_board_right(board):
    old_board = [row.copy() for row in board]
    for col in range(len(board)-2, -1, -1):
        if not any([r[col] for r in board]): continue
        for row in range(len(board)):
            if not board[row][col]: continue
            right = board[row][col+1:]
            if not any(right):
                board[row][-1] = board[row][col]
                board[row][col] = None
                continue
            closest = None
            for i, cell in list(enumerate(right))[::-1]:
                if cell:
                    closest = i
                    break
            if board[row][col+closest+1] == board[row][col]:
                board[row][col+closest+1] *= 2
                board[row][col] = None
            elif not board[row][col+closest]:
                board[row][col+closest] = board[row][col]
                board[row][col] = None
    return board, old_board != board

def print_board(board, key=None):
    if key: print(key)
    for row in board:
        for cell in row:
            print(cell if cell else "_", end=" ")
        print()
    print()

def fill_board_random(board):
    row, col = random.randint(0, 3), random.randint(0, 3)
    while board[row][col]:
        row, col = random.randint(0, 3), random.randint(0, 3)
    board[row][col] = 2

def check_neighbours(board):
    for row in range(len(board)):
        for col in range(len(board)):
            current = board[row][col]
            if not current: continue
            if row > 0 and board[row-1][col] == current: return True
            if row < len(board)-1 and board[row+1][col] == current: return True
            if col > 0 and board[row][col-1] == current: return True
            if col < len(board)-1 and board[row][col+1] == current: return True
    return False

print_board(board)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            pressed = event.key
            changed = False
            if pressed in [pygame.K_w, pygame.K_UP]:
                board, changed = move_board_up(board)
            if pressed in [pygame.K_s, pygame.K_DOWN]:
                board, changed = move_board_down(board)
            if pressed in [pygame.K_a, pygame.K_LEFT]:
                board, changed = move_board_left(board)
            if pressed in [pygame.K_d, pygame.K_RIGHT]:
                board, changed = move_board_right(board)
            if changed: fill_board_random(board)
            print_board(board, pressed)

    if not any([any(r) for r in board]):
        fill_board_random(board)

    board_surf = pygame.Surface((board_size+4, board_size+4))
    board_surf.fill((255, 255, 255))
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col]:
                rect_x = cell_size * col + 4
                rect_y = cell_size * row + 4
                rect_width = cell_size - 4
                rect_height = cell_size - 4
                pygame.draw.rect(board_surf, colors[board[row][col]], (rect_x, rect_y, rect_width, rect_height))
                text_surf = font.render(str(board[row][col]), True, (255, 255, 255))
                text_x = rect_x + (rect_width - text_surf.get_width()) / 2
                text_y = rect_y + (rect_height - text_surf.get_height()) / 2
                board_surf.blit(text_surf, (text_x, text_y))
    
    board_surf.set_alpha(200)
    screen.fill((200, 200, 200))
    screen.blit(bg, (0, 0))
    screen.blit(board_surf, ((screen_size-board_size)/2-2, (screen_size-board_size)/2-2))
    pygame.display.flip()

    if all([all(r) for r in board]) and not check_neighbours(board):
        print("Game Over")
        time.sleep(2)
        running = False

    clock.tick(60)

pygame.quit()