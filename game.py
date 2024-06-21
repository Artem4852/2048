import pygame, random, time

screen_size = 600
board_size = 400
tile_size = 100

bg = pygame.image.load("bg.png")

pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

board = [[0 for _ in range(4)] for _ in range(4)]

colors = {
    2: (255, 161, 125, 255),
    4: (255, 205, 125, 255),
    8: (255, 184, 125, 255),
    16: (255, 138, 125, 255),
    32: (255, 220, 125, 255),
    64: (255, 200, 125, 255),
    128: (176, 100, 70, 255),
    256: (191, 145, 71, 255),
    512: (128, 53, 45, 255),
    1024: (217, 178, 72, 255),
    2048: (255, 198, 125, 255),
    4096: (255, 156, 125, 255),
}

def move_board_up(board):
    score = 0
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
            for i, tile in list(enumerate(above))[::-1]:
                if tile:
                    closest = i
                    break
            if board[closest][col] == board[row][col]:
                score += board[closest][col] * 2
                board[closest][col] *= 2
                board[row][col] = None
            elif not board[closest+1][col]:
                board[closest+1][col] = board[row][col]
                board[row][col] = None
    return board, old_board != board, score

def move_board_down(board):
    score = 0
    old_board = [row.copy() for row in board]
    for row in list(range(len(board)-1))[::-1]:
        for col in range(len(board)):
            if not board[row][col]: continue
            below = [r[col] for r in board[row+1:]]
            if not any(below):
                board[-1][col] = board[row][col]
                board[row][col] = None
                continue
            closest = None
            for i, tile in enumerate(below):
                if tile:
                    closest = i
                    break
            if board[row+closest+1][col] == board[row][col]:
                score += board[row+closest+1][col] * 2
                board[row+closest+1][col] *= 2
                board[row][col] = None
            elif not board[row+closest][col]:
                board[row+closest][col] = board[row][col]
                board[row][col] = None
    return board, old_board != board, score

def move_board_left(board):
    score = 0
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
            for i, tile in list(enumerate(left))[::-1]:
                if tile:
                    closest = i
                    break
            if board[row][closest] == board[row][col]:
                score += board[row][closest] * 2
                board[row][closest] *= 2
                board[row][col] = None
            elif not board[row][closest+1]:
                board[row][closest+1] = board[row][col]
                board[row][col] = None
    return board, old_board != board, score

def move_board_right(board):
    score = 0
    old_board = [row.copy() for row in board]
    for col in list(range(len(board)-1))[::-1]:
        for row in range(len(board)):
            if not board[row][col]: continue
            right = board[row][col+1:]
            if not any(right):
                board[row][-1] = board[row][col]
                board[row][col] = None
                continue
            closest = None
            for i, tile in enumerate(right):
                if tile:
                    closest = i
                    break
            if board[row][col+1+closest] == board[row][col]:
                score += board[row][col+closest+1] * 2
                board[row][col+closest+1] *= 2
                board[row][col] = None
            elif not board[row][col+closest]:
                board[row][col+closest] = board[row][col]
                board[row][col] = None
    return board, old_board != board, score

def print_board(board, key=None):
    if key: print(key)
    for row in board:
        for tile in row:
            print(tile if tile else "_", end=" ")
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

def interpolate(start_value, end_value, step, total_steps):
    delta = end_value - start_value
    return start_value + (delta * (step / total_steps))

def get_tiles_positions(board):
    tiles = []
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col]:
                tiles.append = [board[row][col], (row * tile_size, col * tile_size)]
    return tiles

print_board(board)
running = True
score = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            pressed = event.key
            changed = False
            if pressed in [pygame.K_w, pygame.K_UP]:
                board, changed, points = move_board_up(board)
            if pressed in [pygame.K_s, pygame.K_DOWN]:
                board, changed, points = move_board_down(board)
            if pressed in [pygame.K_a, pygame.K_LEFT]:
                board, changed, points = move_board_left(board)
            if pressed in [pygame.K_d, pygame.K_RIGHT]:
                board, changed, points = move_board_right(board)
            if changed: fill_board_random(board); score += points
            pygame.display.set_caption(f"2048 | Score: {score}")
            # print_board(board, pressed)

    if not any([any(r) for r in board]):
        fill_board_random(board)

    board_surf = pygame.Surface((board_size+4, board_size+4), pygame.SRCALPHA)
    board_surf.fill((255, 255, 255, 255))
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col]:
                rect_x = tile_size * col + 4
                rect_y = tile_size * row + 4
                rect_width = tile_size - 4
                rect_height = tile_size - 4
                pygame.draw.rect(board_surf, colors[board[row][col]], (rect_x, rect_y, rect_width, rect_height))
                text_surf = font.render(str(board[row][col]), True, (255, 255, 255, 255))
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
        for i in range(10):
            pygame.display.set_caption(f"2048 | Game Over | Score: {score}")
            pygame.display.flip()
            pygame.time.delay(200)
            pygame.display.set_caption(f"2048 | Score: {score}")
            pygame.display.flip()
            pygame.time.delay(200)
        running = False

    clock.tick(60)

pygame.quit()