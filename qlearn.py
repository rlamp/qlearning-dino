import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np

from game import Game


parser = argparse.ArgumentParser(description='Q-Learning for Chrome Dino run.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--score", type=int, default=0,
                    help="Starting score limit.")
parser.add_argument("--iter", type=int, default=0,
                    help="Starting iteration number.")
parser.add_argument("--qtable", type=str, default=None,
                    help="Path to existing Q-table.")

args = parser.parse_args()

try:
    i_starting = 1
    if args.qtable is not None:
        q_table = np.load(args.qtable)
        if args.iter:
            i_starting = args.iter + 1
    else:
        q_table = np.zeros(
            [Game.observation_space_n, Game.action_space_n], dtype=np.float16)

    score_limit = args.score

    env = Game()

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
            elif (env.get_score() >= score_limit and
                  random.uniform(0, 1) < epsilon):
                action = random.randint(0, 1)  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, game_over = env.take_action(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = ((1 - alpha) * old_value + alpha *
                         (reward + gamma * next_max))

            q_table[state, action] = np.float16(new_value)

            state = next_state

        # Failed at lower score, lower score_limit
        curr_score = env.get_score()
        print(f"{datetime.now()} {i} {score_limit} {curr_score}")
        score_limit = (curr_score // 100) * 100

        if i % 20 == 0:
            score_limit += 100

        if i % 500 == 0:
            np.save(f'q_table_{i}.npy', q_table)

    print("Training finished.")


finally:
    # Always close driver
    np.save(f'q_table_{i}_ctrlc.npy', q_table)
    env.end()
