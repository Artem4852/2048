import pygame, random

bg = pygame.image.load("bg.png")

pygame.init()
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

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

class Game():
    def __init__(self, screen_size = 600, board_size = 400, tile_size = 100, user_mode = True):
        self.screen_size = screen_size 
        self.board_size = board_size
        self.tile_size = tile_size

        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.running = True
        self.score = 0

        self.user_mode = user_mode

        self.restart()

    def restart(self):
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.print_board()
    
    def print_board(self):
        for row in self.board:
            for tile in row:
                print(tile if tile else "_", end=" ")
            print()
        print()

    def make_move(self, direction):
        changed = False
        if direction == "up":
            changed = self.move_board_up()
        if direction == "down":
            changed = self.move_board_down()
        if direction == "left":
            changed = self.move_board_left()
        if direction == "right":
            changed = self.move_board_right()
        if changed: self.fill_board_random()
        pygame.display.set_caption(f"2048 | Score: {self.score}")
        self.print_board()
        return changed

    def move_board_up(self):
        old_board = [row.copy() for row in self.board]
        for row in range(1, len(self.board)):
            if not any(self.board[row]): continue
            for col in range(len(self.board)):
                if not self.board[row][col]: continue
                above = [r[col] for r in self.board[:row]]
                if not any(above):
                    self.board[0][col] = self.board[row][col]
                    self.board[row][col] = None
                    continue
                closest = None
                for i, tile in list(enumerate(above))[::-1]:
                    if tile:
                        closest = i
                        break
                if self.board[closest][col] == self.board[row][col]:
                    self.score += self.board[closest][col] * 2
                    self.board[closest][col] *= 2
                    self.board[row][col] = None
                elif not self.board[closest+1][col]:
                    self.board[closest+1][col] = self.board[row][col]
                    self.board[row][col] = None
        return old_board != self.board

    def move_board_down(self):
        old_board = [row.copy() for row in self.board]
        for row in list(range(len(self.board)-1))[::-1]:
            for col in range(len(self.board)):
                if not self.board[row][col]: continue
                below = [r[col] for r in self.board[row+1:]]
                if not any(below):
                    self.board[-1][col] = self.board[row][col]
                    self.board[row][col] = None
                    continue
                closest = None
                for i, tile in enumerate(below):
                    if tile:
                        closest = i
                        break
                if self.board[row+closest+1][col] == self.board[row][col]:
                    self.score += self.board[row+closest+1][col] * 2
                    self.board[row+closest+1][col] *= 2
                    self.board[row][col] = None
                elif not self.board[row+closest][col]:
                    self.board[row+closest][col] = self.board[row][col]
                    self.board[row][col] = None
        return old_board != self.board

    def move_board_left(self):
        old_board = [row.copy() for row in self.board]
        for col in range(1, len(self.board)):
            if not any([r[col] for r in self.board]): continue
            for row in range(len(self.board)):
                if not self.board[row][col]: continue
                left = self.board[row][:col]
                if not any(left):
                    self.board[row][0] = self.board[row][col]
                    self.board[row][col] = None
                    continue
                closest = None
                for i, tile in list(enumerate(left))[::-1]:
                    if tile:
                        closest = i
                        break
                if self.board[row][closest] == self.board[row][col]:
                    self.score += self.board[row][closest] * 2
                    self.board[row][closest] *= 2
                    self.board[row][col] = None
                elif not self.board[row][closest+1]:
                    self.board[row][closest+1] = self.board[row][col]
                    self.board[row][col] = None
        return old_board != self.board

    def move_board_right(self):
        old_board = [row.copy() for row in self.board]
        for col in list(range(len(self.board)-1))[::-1]:
            for row in range(len(self.board)):
                if not self.board[row][col]: continue
                right = self.board[row][col+1:]
                if not any(right):
                    self.board[row][-1] = self.board[row][col]
                    self.board[row][col] = None
                    continue
                closest = None
                for i, tile in enumerate(right):
                    if tile:
                        closest = i
                        break
                if self.board[row][col+1+closest] == self.board[row][col]:
                    self.score += self.board[row][col+closest+1] * 2
                    self.board[row][col+closest+1] *= 2
                    self.board[row][col] = None
                elif not self.board[row][col+closest]:
                    self.board[row][col+closest] = self.board[row][col]
                    self.board[row][col] = None
        return old_board != self.board

    def fill_board_random(self):
        row, col = random.randint(0, 3), random.randint(0, 3)
        while self.board[row][col]:
            row, col = random.randint(0, 3), random.randint(0, 3)
        self.board[row][col] = 2

    def check_neighbours(self):
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                current = self.board[row][col]
                if not current: continue
                if row > 0 and self.board[row-1][col] == current: return True
                if row < len(self.board)-1 and self.board[row+1][col] == current: return True
                if col > 0 and self.board[row][col-1] == current: return True
                if col < len(self.board)-1 and self.board[row][col+1] == current: return True
        return False

    def render_board(self):
        board_surf = pygame.Surface((self.board_size+4, self.board_size+4), pygame.SRCALPHA)
        board_surf.fill((255, 255, 255, 255))
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col]:
                    rect_x = self.tile_size * col + 4
                    rect_y = self.tile_size * row + 4
                    rect_width = self.tile_size - 4
                    rect_height = self.tile_size - 4
                    pygame.draw.rect(board_surf, colors[self.board[row][col]], (rect_x, rect_y, rect_width, rect_height))
                    text_surf = font.render(str(self.board[row][col]), True, (255, 255, 255, 255))
                    text_x = rect_x + (rect_width - text_surf.get_width()) / 2
                    text_y = rect_y + (rect_height - text_surf.get_height()) / 2
                    board_surf.blit(text_surf, (text_x, text_y))
        board_surf.set_alpha(200)
        self.screen.fill((200, 200, 200))
        self.screen.blit(bg, (0, 0))
        self.screen.blit(board_surf, ((self.screen_size-self.board_size)/2-2, (self.screen_size-self.board_size)/2-2))
        pygame.display.flip()
    
    def lost(self):
        return all([all(r) for r in self.board]) and not self.check_neighbours()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    pressed = event.key
                    self.make_move("up" if pressed in [pygame.K_w, pygame.K_UP] else "down" if pressed in [pygame.K_s, pygame.K_DOWN] else "left" if pressed in [pygame.K_a, pygame.K_LEFT] else "right" if pressed in [pygame.K_d, pygame.K_RIGHT] else None)
                pygame.display.set_caption(f"2048 | Score: {self.score}")
            
            if not any([any(r) for r in self.board]):
                self.fill_board_random()
            
            self.render_board()

            if self.lost():
                print("Game Over")
                for _ in range(10):
                    pygame.display.set_caption(f"2048 | Game Over | Score: {self.score}")
                    pygame.display.flip()
                    pygame.time.delay(200)
                    pygame.display.set_caption(f"2048 | Score: {self.score}")
                    pygame.display.flip()
                    pygame.time.delay(200)
                self.running = False
            
            clock.tick(60)
        
        pygame.quit()