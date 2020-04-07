import pygame
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from matplotlib import style


class SquareMovement:
    def __init__(self, dist=15, timeout=30):
        self._now = -1
        self.now = -1
        self.dist = dist
        self.A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                  1, 1, 1, 1, 1, 2]
        self.timeout = len(self.A)

    def __next__(self):
        self._now += 1
        if not self._now % self.dist:
            self.now = (self.now + 1) % 4

        if self._now >= self.timeout:
            return 0
        return self.A[self._now]

    def __repr__(self):
        return f"Square: {self.now}"


class Game:
    _count = 0

    def __init__(self, width=1.4e3, height=8e2, render=False, food_ammount=3,
                 view_len=4, time_out=10000):
        if render:
            pygame.init()
            width = int(width)
            height = int(height)
            self.screen = pygame.display.set_mode((width, height))

        self.render = render
        self.food_on_screen = food_ammount
        self.view_len = view_len
        self.time_out = time_out

        self.size = self.width, self.height = width, height
        self.score = 0
        self.direction = 1
        self.x = 0
        self.y = 0
        self.tail = [[self.x, self.y]]
        self.tail_len = 10
        self.food = []
        self.done = False
        self.current_time = 0

        self.square_move = SquareMovement()
        self.move_time = 1

        self.rect_size = 25
        self.speed_multiplier = 1
        self._reset()
        Game._count += 1

    def __del__(self):
        Game._count -= 1
        if Game._count < 1:
            pygame.quit()

    def _reset(self):
        self.score = 0
        self.direction = 1
        self.speed_multiplier = 1
        self.x = (self.width / 2) // self.rect_size * self.rect_size
        self.y = (self.height / 2) // self.rect_size * self.rect_size
        self.tail_len = 10
        self.tail = [[self.x, self.y]]
        self.food = []
        self.done = False
        self.current_time = 0
        self.food_refill(True, self.food_on_screen)

    def check_border(self, border=True):
        hit = False
        if border:  # Screen edge is border
            # Checking X values in proper range
            if self.x >= self.width:

                self.x = self.width - self.rect_size
                hit = True
            elif self.x < 0:
                self.x = 0
                hit = True
            # Checking Y values in proper range
            if self.y >= self.height:
                self.y = self.height - self.rect_size
                hit = True
            elif self.y < 0:
                self.y = 0
                hit = True  # if colision with wall return hit

        else:  # Screen edge is borderless (Continuous world)
            # Checking X values in proper range
            if self.x >= self.width:
                self.x = 0
            elif self.x < 0:
                self.x = self.width - self.rect_size
            # Checking Y values in proper range
            if self.y >= self.height:
                self.y = 0
            elif self.y < 0:
                self.y = self.height - self.rect_size
        return hit

    def check_collision_with_food(self, x, y):
        for pos in self.food:
            if [x, y] == pos:
                return True
        return False

    def check_collision_with_observationtacles(self, x, y):
        for pos in self.tail[1:-1]:
            if [x, y] == pos:
                # self.speed_multiplier = 0
                return True
        pass

    def check_collision_with_snake(self, x, y):
        for pos in self.tail:
            if [x, y] == pos:
                return True
        return False

    def display_score(self):
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render('Score = ' + str(self.score), False, (255, 255, 255))
        self.screen.blit(text_surface, (0, 0))

    def draw(self):
        if self.done:
            head_color = (255, 0, 0)
        else:
            head_color = (130, 255, 255)
        self.screen.fill((30, 30, 50))
        pygame.draw.rect(self.screen, (50, 150, 130),
                         (self.tail[0][0], self.tail[0][1], self.rect_size,
                          self.rect_size))  # last tail piece
        for tail in self.tail[1:]:
            pygame.draw.rect(self.screen, (35, 120, 50), (tail[0], tail[1], self.rect_size, self.rect_size))
        pygame.draw.rect(self.screen, head_color, (self.x, self.y, self.rect_size, self.rect_size))

        for food in self.food:
            pygame.draw.rect(self.screen, (0, 255, 0), (food[0], food[1], self.rect_size, self.rect_size))

        self.display_score()
        pygame.display.update()

        # surf = pygame.display.get_surface()
        # pygame.image.save(surf, f'image{index}.png')

    def eat_food(self):
        """

        Returns
        -------
        int: quantity of eaten food
        """
        out = 0
        for i, food in enumerate(self.food):
            if self.check_collision_with_snake(food[0], food[1]):
                self.food.pop(i)
                self.tail_len += 1
                self.score += 1
                out += 1
                self.place_food()
                break
        return out

    def food_refill(self, multi_food=False, n=1):
        while multi_food and len(self.food) < n \
                or len(self.food) < 1:
            self.place_food()

    def move_snake(self, new_direction, reverse=False):
        """
        Directions
          0       Up
        3   1     Left / Right
          2       Down
        """
        new_direction = int(new_direction)
        current_speed = self.speed_multiplier * self.rect_size
        # Turn only left or right
        if not ((self.direction in [0, 2] and new_direction in [0, 2])
                or (self.direction in [1, 3] and new_direction in [1, 3])):
            self.direction = new_direction

        if reverse:
            if self.direction == 0:
                self.y += current_speed
            elif self.direction == 1:
                self.x -= current_speed
            elif self.direction == 2:
                self.y -= current_speed
            else:
                self.x += current_speed
        else:
            if self.direction == 0:
                self.y -= current_speed
            elif self.direction == 1:
                self.x += current_speed
            elif self.direction == 2:
                self.y += current_speed
            else:
                self.x -= current_speed

    def observation_area(self):
        depth = self.view_len * 2 + 1  # in to direction + 1(center)
        view_area = np.zeros((depth, depth), dtype=int)
        for iy in range(depth):
            for ix in range(depth):
                x = int(self.x + (ix - self.view_len) * self.rect_size)
                y = int(self.y + (iy - self.view_len) * self.rect_size)  # Y is drawn from top to bot

                if x > self.width or x < 0 or \
                        y >= self.height or y < 0:
                    view_area[iy, ix] = 2
                    continue

                for f in self.food:
                    if [x, y] == f:
                        view_area[iy, ix] = 1
                        break

                for tail in self.tail:
                    if [x, y] == tail:
                        view_area[iy, ix] = 2
                        break

        return self.direction, view_area

    def observation(self):
        """

        :return:
        int:
            direction in range<0, 3>
        list:
            food-head relative position shape=(2)
        list:
            view-area, shape(3,3), elements: 0-path, 1-collision
        """
        view_area = np.zeros((3, 3), dtype=int)
        for iy in range(3):
            for ix in range(3):
                x = int(self.x + (ix - 1) * self.rect_size)
                y = int(self.y + (iy - 1) * self.rect_size)

                if x > self.width or x < 0 or \
                        y >= self.height or y < 0:
                    view_area[iy, ix] = 1
                    continue

                for tail in self.tail:
                    if [x, y] == tail:
                        view_area[iy, ix] = 1
                        break

        if len(self.food) > 0:
            food = self.food[0]
        else:
            food = [2, 2]
            print(f"No Food info")
            return self.direction, food, view_area

        out = [food[0] - self.x, food[1] - self.y]
        return self.direction, out, view_area

    def place_food(self):
        while True:
            rx = np.random.rand() * (self.width - 1)
            ry = np.random.rand() * (self.height - 1)

            rx = round(rx // self.rect_size * self.rect_size)
            ry = round(ry // self.rect_size * self.rect_size)

            if self.check_collision_with_food(rx, ry):
                continue
            break
        self.food += [[rx, ry]]

    # def play(self, delay=0.03):
    #     valid = True
    #     try:
    #         index = 1
    #         while valid:
    #             valid, reward, state = self.step()
    #
    #             if self.render:
    #                 self.draw()
    #                 time.sleep(delay)
    #             index += 1
    #     finally:
    #         if self.render:
    #             pygame.display.quit()

    def reset(self):
        """
        Returns
        -------
        tuple: observation
            int: direction
            list of ints: shape=(2), food position relative,
            list of ints: shape=(3,3), view_area
                value 0 is path,
                value 1 is wall / tail
        """
        self._reset()
        return self.observation()

    def step(self, action):
        """
        Function makes one step in game and returns new observation
        Parameters
        ----------
        action: int
            0 - turn left
            1 - go straight
            2 - turn right

        Returns
        -------
        tuple:
            bool: continue_game
            int: reward
            tuple: observation
                int: direction
                list of ints: shape=(2), food position relative,
                list of ints: shape=(3,3), view_area
                    value 0 is path,
                    value 1 is wall / tail
        """
        new_direction = (self.direction + (action - 1)) % 4
        if self.done:
            print("Run has ended")
        f_run = True
        self.current_time += 1
        self.food_refill(True, self.food_on_screen)

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    f_run = False
                    break

        self.move_snake(new_direction)
        hit1 = self.check_border()
        hit2 = self.check_collision_with_observationtacles(self.x, self.y)
        if hit1 or hit2:  # <<---- Collision, Disable loop
            f_run = False
            # if hit2:
            #     self.move_snake(reverse=True)
            self.speed_multiplier = 0
            _color = (255, 0, 0)
        else:
            _color = (130, 255, 255)

        self.update_tail()
        reward = self.eat_food() * 50 - 1  # Eaten food is worth 5
        observation = self.observation()

        if not f_run:  # Devalue reward
            self.done = True
            reward = -100

        if self.current_time >= self.time_out:
            f_run = False
            print(f"Timeout! score: {self.score}")
            self.done = True

        return f_run, reward, observation

    def update_tail(self):
        if self.tail[-1] != [self.x, self.y]:
            self.tail += [[self.x, self.y]]

        if len(self.tail) > (self.tail_len + 1):
            self.tail = self.tail[-self.tail_len - 1:]
            # self.tail.pop(0)


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
    tuple: index slice to q-vals
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


"Train"
EPISODES = 50_000
SHOW_EVERY = EPISODES - 10 // 3
LEARNING_RATE = 0.06
DISCOUNT = 0.9

"Environment"
ACTIONS = 3  # 4 Moves possible
MOVE_DIRECTIONS = 4  # state movement directions
FIELD_STATES = 2
VIEW_AREA = 9
FOOD_SIZE = [3, 3]

"Exploration"
eps = 0.5
EPS_OFFSET = 0.005
EPS_START_DECAYING = 0
EPS_DECAY_AT = EPISODES // 5
eps_iterator = iter(np.linspace(eps, 0, EPS_DECAY_AT - EPS_START_DECAYING))

"Q-Table initialization"
size = [MOVE_DIRECTIONS] + FOOD_SIZE + [FIELD_STATES ** VIEW_AREA] + [ACTIONS]
try:
    q_table = np.load('last_qtable.npy', allow_pickle=True)
except FileNotFoundError:
    print(f"Creating new qtable!")
    q_table = np.random.uniform(-2, -1, size=size)

try:
    stats = np.load('last_stats.npy', allow_pickle=True).item()
    episode_offset = stats['episode'][-1] + 1
except FileNotFoundError:
    stats = {
        "episode": [],
        "eps": [],
        "score": [],
        "food_eaten": []
    }
    episode_offset = 0

for episode in range(0 + episode_offset, EPISODES + episode_offset):
    # Show very and show last
    if not episode % SHOW_EVERY or episode >= EPISODES + episode_offset - 1:
        render = True
    else:
        render = False
    # if :

    game = Game(food_ammount=1, render=render)
    valid = True
    observation = Game().reset()
    score = 0

    if EPS_DECAY_AT + episode_offset > episode > EPS_START_DECAYING + episode_offset:
        eps = next(eps_iterator) + EPS_OFFSET
    else:
        eps = EPS_OFFSET
    if render:
        print(" = = New game = = "*3)
    while valid:
        q_values = get_discrete_vals(q_table, observation)
        if render:
            print(f"Direction: {observation[0]}, food: {observation[1]}, q_vals: {q_values}")
            print(observation[-1])

        if eps > np.random.random():
            action = np.random.randint(0, ACTIONS)
        else:
            action = np.argmax(q_values)

        old_q = q_values[action]

        valid, reward, observation = game.step(action=action)
        max_future_q = max(get_discrete_vals(q_table, observation))
        new_q = (1 - LEARNING_RATE) * old_q \
            + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state_index(observation) + (action,)] = new_q
        if render:
            game.draw()
            # print(f"\tFuture q: {get_discrete_vals(q_table, observation)}")
            time.sleep(0.01)
            print(f"Q: {old_q:>2.4f}, new_q {new_q:>2.4}, reward: {reward:>3}")
        score += reward

    stats['episode'].append(episode)
    stats['eps'].append(eps)
    stats['score'].append(score)
    stats['food_eaten'].append(game.score)

    if game.score > 0:
        print(f"Ep[{episode:^7}], food_eaten:{game.score:>4}, Eps: {eps:>1.3f}, reward:{score:>6}")
    # if render:
    #     input("Waiting...")

# Saving outputs
os.makedirs('graphs', exist_ok=True)

np.save('last_qtable.npy', q_table)
np.save('last_stats.npy', stats)

pygame.quit()

style.use('ggplot')
plt.scatter(stats['episode'][episode_offset:], stats['score'][episode_offset:], alpha=0.13, marker='s', edgecolors='m',
            label="Score")
plt.legend(loc=3)
plt.savefig(f"graphs/rewards-{episode_offset}-{episode_offset + EPISODES - 1}")
plt.show()
