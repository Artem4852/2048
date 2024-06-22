from game import Game
from model import DinoGamer
import time, pygame, json

actions = ["up", "down", "left", "right"]

agent = DinoGamer()
load_model = True

with open("stats.json", "r") as f: stats = json.load(f)

if load_model: agent.model.load_model("model.pth")

else:
    stats["games"] = 0
    stats["moves"] = [0, 0]
    stats["epsilon"] = agent.epsilon

agent.epsilon = stats["epsilon"]

time.sleep(1)

quit_train = False

while not quit_train:
    game = Game(user_mode=False)
    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_train = True
        if quit_train: break
        state = game.get_state()
        action, random_choice = agent.select_action(state)
        stats["moves"][0 if random_choice else 1] += 1
        next_state, reward = game.make_move(actions[action])
        game.render_board(stats["moves"], stats["games"], reward)
        game.clock.tick(1)
        agent.train(state, action, reward, next_state, not game.running)
    if stats["games"] % 10 == 0: agent.model.save_model(f"model.pth")
    stats["games"] += 1
    stats["max_score"] = game.score if game.score > stats["max_score"] else stats["max_score"]
    stats["epsilon"] = agent.epsilon
    with open("stats.json", "w") as f: json.dump(stats, f)
    pygame.quit()