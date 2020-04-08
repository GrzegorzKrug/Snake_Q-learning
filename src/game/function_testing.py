# from .snake import discrete_state_index
import numpy as np

FIELD_STATES = 2


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
        int: view area index <0, field_vals**(fields-1)>
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


def test0():
    direction = 0
    food_x = 0
    food_y = 0
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 1, 1, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out


def test1():
    direction = 1
    food_x = -1
    food_y = 1
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 0, 2, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out


def test2():
    direction = 1
    food_x = 1
    food_y = 0
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 2, 1, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out


def test3():
    direction = 1
    food_x = 1
    food_y = -1
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 2, 0, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out


def test4():
    direction = 1
    food_x = 110
    food_y = -200
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 2, 0, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out


def test5_float():
    direction = 1
    food_x = -110.0
    food_y = 25.0
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 0, 2, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out


def test6_float():
    direction = 1
    food_x = -110.0
    food_y = 0.0
    area = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]])
    observation = (direction, [food_x, food_y], area)
    good = (direction, 0, 1, 2**4)
    out = discrete_state_index(observation=observation)
    assert good == out
