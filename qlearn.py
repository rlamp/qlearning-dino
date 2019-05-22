import argparse
import os
import pickle
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
        with open(args.qtable, 'rb') as f:
            q_table = pickle.load(f)
        if args.iter:
            i_starting = args.iter + 1
    else:
        q_table = {}
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
                # Explore action space
                action = random.randint(0, 1)
            else:
                # Exploit learned values
                action = np.argmax(q_table.get(state, [0, 0]))

            next_state, reward, game_over = env.take_action(action)

            try:
                old_value = q_table.get(state, [0, 0])[action]
            except:
                old_value = 0
            next_max = np.max(q_table.get(next_state, [0, 0]))

            new_value = ((1 - alpha) * old_value + alpha *
                         (reward + gamma * next_max))
            new_value = np.float16(new_value)

            try:
                q_table[state][action] = new_value
            except:
                new_arr = np.array([0, 0], dtype=np.float16)
                new_arr[action] = new_value
                q_table[state] = new_arr

            state = next_state

        # Failed at lower score, lower score_limit
        curr_score = env.get_score()
        print(f"{datetime.now()} {i} {score_limit} {curr_score}")
        score_limit = (curr_score // 100) * 100

        if i % 20 == 0:
            score_limit += 100

        if i % 500 == 0:
            sparse.save_npz(f'q_table_{i}.npz', q_table)
            with open(f'q_table_{i}.pickle', 'wb') as f:
                pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training finished.")


finally:
    # Always close driver
    with open(f'q_table_{i}_ctrlc.pickle', 'wb') as f:
        pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    env.end()
