import argparse

import numpy as np

from game import Game

parser = argparse.ArgumentParser(description='Q-Learning for Chrome Dino run.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("qtable", type=str,
                    help="Path to Q-table.")

args = parser.parse_args()


env = Game()

try:
    q_table = np.load(args.qtable)

    while True:
        state = env.reset()
        game_over = False

        while not game_over:
            # Cant jump while jumping
            if state % 2 == 1:
                action = 0
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            state, _, game_over = env.take_action(action)

finally:
    # Always close driver
    env.end()
