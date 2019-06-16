import argparse
import pickle

import numpy as np

from game import Game

parser = argparse.ArgumentParser(description='Play Chrome Dino run.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--no-acceleration", action='store_true',
                    help="No acceleration of game.")
parser.add_argument("qtable", type=str,
                    help="Path to Q-table.")

args = parser.parse_args()

with open(args.qtable, 'rb') as f:
    q_table = pickle.load(f)

try:
    env = Game(no_acceleration=args.no_acceleration)
    env.start()

    while True:
        state = env.reset()
        game_over = False

        while not game_over:
            # Cant jump while jumping
            if state % 2 == 1:
                action = 0
            else:
                # Exploit learned values
                action = np.argmax(q_table.get(state, [0, 0]))

            state, _, game_over = env.take_action(action)

finally:
    # Always close driver
    env.end()
