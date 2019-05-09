import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np

from game import Game


def xprint(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=sys.stderr, **kwargs)


parser = argparse.ArgumentParser(description='Q-Learning for Chrome Dino run.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--iter", type=int, default=0,
                    help="Starting iteration number.")
parser.add_argument("--qtable", type=str, default=None,
                    help="Path to existing Q-table.")
parser.add_argument("--silent", dest='verbose', action='store_false',
                    help="Do not print states and rewards.")

args = parser.parse_args()


env = Game()

try:
    i_starting = 1
    if args.qtable is not None:
        q_table = np.load(args.qtable)
        if args.iter:
            i_starting = args.iter + 1
    else:
        q_table = np.zeros(
            [env.observation_space_n, env.action_space_n], dtype=np.float16)

    score_limit = 0

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
            elif (random.uniform(0, 1) < epsilon and
                  env.get_score() >= score_limit):
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

            q_table[state, action] = np.float16(new_value)

            state = next_state

        xprint(f"{datetime.now()} Episode: {i}")
        if i % 200 == 0:
            # Evercleay 200 episodes test if it can pass the 100 score section
            # 10 times in a row
            test_passed = True
            for testn in range(10):
                state = env.reset()
                game_over = False

                while not game_over:
                    # Exploit learned values
                    action = np.argmax(q_table[state])
                    state, _, game_over = env.take_action(action)

                if env.get_score() < score_limit + 100:
                    xprint(f'Test {testn} failed.')
                    test_passed = False
                    break

            if test_passed:
                score_limit = score_limit + 100
                xprint(f'Test {score_limit} passed')

        if i % 500 == 0:
            # old_file = f'q_table_{i-50}.npy'
            # if os.path.isfile(old_file):
            #     os.remove(old_file)
            np.save(f'q_table_{i}.npy', q_table)

    xprint("Training finished.")


finally:
    # Always close driver
    np.save(f'q_table_{i}_ctrlc.npy', q_table)
    env.end()
