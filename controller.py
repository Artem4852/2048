from game import Game
from model import DinoGamer

actions = ["up", "down", "left", "right"]

game = Game(user_mode=False)
agent = DinoGamer()

while game.running:
    state = game.get_state()
    action = agent.select_action(state)
    next_state, reward = game.make_move(actions[action])
    agent.train(state, action, reward, next_state, not game.running)