import argparse
import os
import random
from datetime import datetime

import numpy as np

from game import Game

parser = argparse.ArgumentParser(description='Q-Learning for Chrome Dino run.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--iter", type=int, default=0,
                    help="Starting iteration number.")
parser.add_argument("--qtable", type=str, default=None,
                    help="Path to existing Q-table.")
parser.add_argument("--verbose", action='store_true',
                    help="Print states and rewards.")

args = parser.parse_args()


env = Game()

try:
    i_starting = 1
    if args.qtable is not None:
        q_table = np.load(args.qtable)
        if args.iter:
            i_starting = args.iter + 1
    else:
        q_table = np.zeros([env.observation_space_n, env.action_space_n])

    # Hyperparameters
    alpha = 0.2
    gamma = 0.5
    epsilon = 0.15

    for i in range(i_starting, 100001):
        state = env.reset()

        reward = 0
        game_over = False

        while not game_over:
            # Cant jump while jumping
            if state % 2 == 1:
                action = 0
            elif random.uniform(0, 1) < epsilon:
                action = random.randint(0, 1)  # Explore action space
                if args.verbose and action:
                    print('Random jump')
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, game_over = env.take_action(action)
            if args.verbose:
                print(f'{next_state}\t{reward}')

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = ((1 - alpha) * old_value + alpha *
                         (reward + gamma * next_max))
            q_table[state, action] = new_value

            state = next_state

        print(f"{datetime.now()} Episode: {i}")
        if i % 200 == 0:
            # old_file = f'q_table_{i-50}.npy'
            # if os.path.isfile(old_file):
            #     os.remove(old_file)
            np.save(f'q_table_{i}.npy', q_table)

    print("Training finished.")


finally:
    # Always close driver
    np.save(f'q_table_{i}_ctrlc.npy', q_table)
    env.end()
