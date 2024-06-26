import pygame, random, time
import numpy as np

bg = pygame.image.load("bg.png")


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
        pygame.init()
        self.screen_size = screen_size 
        self.board_size = board_size
        self.tile_size = tile_size

        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.running = True

        self.user_mode = user_mode

        self.restart()

    def restart(self):
        self.score = 0
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.fill_board_random()
        self.print_board()
        self.render_board()

        self.useless_moves = 0
        self.last_unchanged = False
    
    def print_board(self, move=None):
        if move: print("Move:", move)
        for row in self.board:
            for tile in row:
                print(tile if tile else "_", end="  ")
            print()
        print()

    def make_move(self, direction):
        reward = 0
        prev_score = self.score
        prev_top_left_score = self.calculate_top_left_reward()
        prev_max_tile = max([max(row) for row in self.board])
        prev_empty_tiles = sum([row.count(0) for row in self.board])
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

        # reward part
        new_top_left_score = self.calculate_top_left_reward()
        new_max_tile = max([max(row) for row in self.board])
        new_empty_tiles = sum([row.count(0) for row in self.board])

        reward = self.score - prev_score

        top_left_improvement = new_top_left_score - prev_top_left_score
        reward += top_left_improvement * 10

        if new_max_tile > prev_max_tile: reward += 10 * (new_max_tile - prev_max_tile)
        if new_empty_tiles < prev_empty_tiles: reward -= 2 * (prev_empty_tiles - new_empty_tiles)

        if not changed: 
            reward -= 10
            if self.last_unchanged:
                self.useless_moves += 1
                reward -= self.useless_moves * 5

        self.last_unchanged = not changed
        # ---

        self.print_board(direction)
        if self.lost():
            self.running = False
            return self.get_state(), -100
        return self.get_state(), reward

    def move_board_up(self):
        old_board = [row.copy() for row in self.board]
        for row in range(1, len(self.board)):
            if not any(self.board[row]): continue
            for col in range(len(self.board)):
                if not self.board[row][col]: continue
                above = [r[col] for r in self.board[:row]]
                if not any(above):
                    self.board[0][col] = self.board[row][col]
                    self.board[row][col] = 0
                    continue
                closest = 0
                for i, tile in list(enumerate(above))[::-1]:
                    if tile:
                        closest = i
                        break
                if self.board[closest][col] == self.board[row][col]:
                    self.score += self.board[closest][col] * 2
                    self.board[closest][col] *= 2
                    self.board[row][col] = 0
                elif not self.board[closest+1][col]:
                    self.board[closest+1][col] = self.board[row][col]
                    self.board[row][col] = 0
        return old_board != self.board

    def move_board_down(self):
        old_board = [row.copy() for row in self.board]
        for row in list(range(len(self.board)-1))[::-1]:
            for col in range(len(self.board)):
                if not self.board[row][col]: continue
                below = [r[col] for r in self.board[row+1:]]
                if not any(below):
                    self.board[-1][col] = self.board[row][col]
                    self.board[row][col] = 0
                    continue
                closest = 0
                for i, tile in enumerate(below):
                    if tile:
                        closest = i
                        break
                if self.board[row+closest+1][col] == self.board[row][col]:
                    self.score += self.board[row+closest+1][col] * 2
                    self.board[row+closest+1][col] *= 2
                    self.board[row][col] = 0
                elif not self.board[row+closest][col]:
                    self.board[row+closest][col] = self.board[row][col]
                    self.board[row][col] = 0
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
                    self.board[row][col] = 0
                    continue
                closest = 0
                for i, tile in list(enumerate(left))[::-1]:
                    if tile:
                        closest = i
                        break
                if self.board[row][closest] == self.board[row][col]:
                    self.score += self.board[row][closest] * 2
                    self.board[row][closest] *= 2
                    self.board[row][col] = 0
                elif not self.board[row][closest+1]:
                    self.board[row][closest+1] = self.board[row][col]
                    self.board[row][col] = 0
        return old_board != self.board

    def move_board_right(self):
        old_board = [row.copy() for row in self.board]
        for col in list(range(len(self.board)-1))[::-1]:
            for row in range(len(self.board)):
                if not self.board[row][col]: continue
                right = self.board[row][col+1:]
                if not any(right):
                    self.board[row][-1] = self.board[row][col]
                    self.board[row][col] = 0
                    continue
                closest = 0
                for i, tile in enumerate(right):
                    if tile:
                        closest = i
                        break
                if self.board[row][col+1+closest] == self.board[row][col]:
                    self.score += self.board[row][col+closest+1] * 2
                    self.board[row][col+closest+1] *= 2
                    self.board[row][col] = 0
                elif not self.board[row][col+closest]:
                    self.board[row][col+closest] = self.board[row][col]
                    self.board[row][col] = 0
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

    def count_merge_possibilities(self):
        count = 0
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                current = self.board[row][col]
                if not current: continue
                if row > 0 and self.board[row-1][col] == current: count += 1
                if row < len(self.board)-1 and self.board[row+1][col] == current: count += 1
                if col > 0 and self.board[row][col-1] == current: count += 1
                if col < len(self.board)-1 and self.board[row][col+1] == current: count += 1
        return count

    def calculate_smoothness(self):
       smoothness = 0
       for i in range(4):
           for j in range(4):
               if j < 3:
                   smoothness -= abs(self.board[i][j] - self.board[i][j+1])
               if i < 3:
                   smoothness -= abs(self.board[i][j] - self.board[i+1][j])
       return smoothness

    def calculate_top_left_reward(self):
        reward = 0
        max_reward = sum([16 ** (4-i) for i in range(4)])  

        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0: continue
                distance = row+col
                value = np.log2(self.board[row][col])
                reward += value * (16 ** (3 - distance))
        return reward / max_reward * 100

    def render_board(self, moves = [0, 0], games = 0, reward = 0):
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
                    text_surf = self.font.render(str(self.board[row][col]), True, (255, 255, 255, 255))
                    text_x = rect_x + (rect_width - text_surf.get_width()) / 2
                    text_y = rect_y + (rect_height - text_surf.get_height()) / 2
                    board_surf.blit(text_surf, (text_x, text_y))
        board_surf.set_alpha(200)
        self.screen.fill((200, 200, 200))
        self.screen.blit(bg, (0, 0))
        self.screen.blit(board_surf, ((self.screen_size-self.board_size)/2-2, (self.screen_size-self.board_size)/2-2))
        pygame.display.set_caption(f"2048 | Score: {self.score} | Games: {games} | Moves: {moves[0]}/{moves[1]} | Reward: {reward}")
        pygame.display.flip()
    
    def lost(self):
        return all([all(r) for r in self.board]) and not self.check_neighbours()

    def get_state(self):
        state = [tile for row in self.board for tile in row]
        log_state = [np.log2(tile) if tile else 1 for tile in state]
        empty_tiles = state.count(0)
        max_tile = max(state)
        merge_possibilities = self.count_merge_possibilities()
        smoothness = self.calculate_smoothness()
        return log_state + [empty_tiles, max_tile, merge_possibilities, smoothness, self.useless_moves]

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    pressed = event.key
                    self.make_move("up" if pressed in [pygame.K_w, pygame.K_UP] else "down" if pressed in [pygame.K_s, pygame.K_DOWN] else "left" if pressed in [pygame.K_a, pygame.K_LEFT] else "right" if pressed in [pygame.K_d, pygame.K_RIGHT] else 0)
                pygame.display.set_caption(f"2048 | Score: {self.score}")
            
            if not any([any(r) for r in self.board]):
                self.fill_board_random()
            
            self.render_board()

            if self.lost():
                print("Game Over")
                time.sleep(2)
                self.running = False
            
            self.clock.tick(60)
        
        pygame.quit()