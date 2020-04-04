import numpy as np
import pytest

FIELD_STATES = 3
VIEW_DISTANCE = 3
AREA = [VIEW_DISTANCE*2+1] * 2
# AREA = AREA[0] ** 2
print("AREA:", AREA)
Q_AREA = (AREA[0] ** 2) ** FIELD_STATES


def discrete_state(q_table, direction, area):
    dc = 0
    print(area)
    for i, field in enumerate(area.ravel()):
        if not field:
            continue
        # if i >= side_a:
        #     i
        add = (FIELD_STATES ** i) * field
        print(f"({FIELD_STATES} ** {i}) - 1 + {field} = {add}")

        dc += add

    # dc_state = q_table[direction, dc, :]
    print(f"Final: {dc}")
    dc_state = None
    return dc_state, int(dc)


size = [4] + [Q_AREA] + [4]
print(f"Size: {size}")

zeros = np.zeros(AREA)
q_table = np.random.uniform(-5, 5, size=size)
print(zeros.shape)
print(q_table.shape)


def test0():
    area0 = zeros.copy()
    _, st = discrete_state(q_table=q_table, direction=0, area=area0)
    assert st == 0


def test1():
    area1 = zeros.copy()
    area1[0, 0] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area1)
    assert st == 1


def test2():
    area2 = zeros.copy()
    area2[0, 0] = 2
    _, st = discrete_state(q_table=q_table, direction=0, area=area2)
    assert st == 2


def test3():
    area4 = zeros.copy()
    area4[0, 1] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area4)
    assert st == 3


def test4():
    area5 = zeros.copy()
    area5[0, 1] = 1
    area5[0, 0] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area5)
    assert st == 4


def test5():
    area6 = zeros.copy()
    area6[0, 0] = 2
    area6[0, 1] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area6)
    assert st == 5


def test6():
    area6 = zeros.copy()
    area6[0, 0] = 0
    area6[0, 1] = 2
    _, st = discrete_state(q_table=q_table, direction=0, area=area6)
    assert st == 6


def test7():
    area7 = zeros.copy()
    area7[0, 0] = 1
    area7[0, 1] = 2
    _, st = discrete_state(q_table=q_table, direction=0, area=area7)
    assert st == 7


def test8():
    area7 = zeros.copy()
    area7[0, 0] = 2
    area7[0, 1] = 2
    _, st = discrete_state(q_table=q_table, direction=0, area=area7)
    assert st == 8


def test9():
    area7 = zeros.copy()
    area7[0, 2] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area7)
    assert st == 9


def test10():
    area7 = zeros.copy()
    area7[0, 0] = 1
    area7[0, 2] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area7)
    assert st == 10


def test2187():
    area8 = zeros.copy()
    area8[1, 0] = 1
    _, st = discrete_state(q_table=q_table, direction=0, area=area8)
    assert st == 2187


def test15207():
    # 020212202
    area8 = zeros.copy()
    area8[0, 1] = 2
    area8[0, 3] = 2
    area8[0, 4] = 1
    area8[0, 5] = 2
    area8[0, 6] = 2
    area8[1, 1] = 2
    _, st = discrete_state(q_table=q_table, direction=0, area=area8)
    assert st == 15207


