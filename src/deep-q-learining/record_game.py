from snake import Game
import numpy as np
import sys
import os
import time
import pygame

FIELD_STATES = 2
FILE = "last_qtable_0"


def get_discrete_vals(q_table, observation):
    """

    Parameters
    ----------
    q_table
    observation

    Returns
    -------
    list: q-values
    """
    index = discrete_state_index(observation)
    q_values = q_table[tuple(index)]
    return q_values


def discrete_state_index(observation):
    """

    Parameters
    ----------
    q_table
    observation

    Returns
    -------
    tuple: index slice to q-vals list
        int: direction
        int: food x relative <0, 2>
        int: food y relative <0, 2>
        int: view area index <0, field_vals**fields>
    """
    direction, food_relative, view_area = observation
    discrete_index = 0

    f_x = int(0 if not food_relative[0] else 1 * np.sign(food_relative[0])) + 1  # Select food relative x
    f_y = int(0 if not food_relative[1] else 1 * np.sign(food_relative[1])) + 1  # Select food relative y
    for i, field in enumerate(view_area.ravel()):
        if not field:  # Ignore 0=Path
            continue

        add = (FIELD_STATES ** i) * field
        discrete_index += add

    return direction, f_x, f_y, discrete_index


if __name__ == "__main__":

    game = Game(food_ammount=1, render=True)
    valid = True
    observation = Game().reset()
    score = 0
    q_table = np.load(f"{FILE}.npy", allow_pickle=True)

    os.makedirs(f"{FILE}", exist_ok=True)
    step = 0
    while valid:
        game.draw()
        surface = pygame.display.get_surface()
        pygame.image.save(surface, f"{FILE}/image_{step}.png")

        old_observation = observation
        current_q_values = get_discrete_vals(q_table, old_observation)

        action = np.argmax(current_q_values)

        old_q = current_q_values[action]

        valid, reward, observation = game.step(action=action)


        step += 1
        # time.sleep(0.03)

    game.draw()
    surface = pygame.display.get_surface()
    pygame.image.save(surface, f"{FILE}/image_{step}.png")
