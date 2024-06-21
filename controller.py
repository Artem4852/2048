from game import Game
from model import DinoGamer
import time, pygame

actions = ["up", "down", "left", "right"]

agent = DinoGamer()

time.sleep(1)

games = 0

quit_train = False

moves = [0, 0]

while not quit_train:
    game = Game(user_mode=False)
    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_train = True
        state = game.get_state()
        action, random_choice = agent.select_action(state)
        moves[0 if random_choice else 1] += 1
        next_state, reward = game.make_move(actions[action])
        game.render_board(moves, games)
        game.clock.tick(120)
        agent.train(state, action, reward, next_state, not game.running)
    games += 1
    pygame.quit()