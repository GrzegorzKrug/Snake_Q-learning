import pygame
import numpy as np
import time


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
                 view_len=4):
        if render:
            pygame.init()
            width = int(width)
            height = int(height)
            self.screen = pygame.display.set_mode((width, height))

        self.render = render
        self.food_on_screen = food_ammount
        self.view_len = view_len

        self.size = self.width, self.height = width, height
        self.score = 0
        self.direction = 1
        self.x = 0
        self.y = 0
        self.tail = [[self.x, self.y]]
        self.tail_len = 10
        self.food = []
        self.done = False

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
        self.done = False
        self.food = []
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

    def check_collision_with_obstacles(self, x, y):
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
        pass

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

    def play(self, delay=0.03):
        valid = True
        try:
            index = 1
            while valid:
                valid, reward, state = self.step()

                if self.render:
                    self.draw()
                    time.sleep(delay)
                index += 1
        finally:
            if self.render:
                pygame.display.quit()

    def reset(self):
        self._reset()
        return self.observation()

    def step(self, new_direction):
        if self.done:
            print("Run has ended")
        f_run = True
        self.food_refill(True, self.food_on_screen)

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    f_run = False
                    break

        self.move_snake(new_direction)
        hit1 = self.check_border()
        hit2 = self.check_collision_with_obstacles(self.x, self.y)
        if hit1 or hit2:  # <<---- Collision, Disable loop
            f_run = False
            # if hit2:
            #     self.move_snake(reverse=True)
            self.speed_multiplier = 0
            _color = (255, 0, 0)
        else:
            _color = (130, 255, 255)

        self.update_tail()
        reward = self.eat_food()
        obs = self.observation()

        if not f_run:  # Devalue reward
            self.done = True
            reward = -10

        return f_run, reward, obs

    def update_tail(self):
        if self.tail[-1] != [self.x, self.y]:
            self.tail += [[self.x, self.y]]

        if len(self.tail) > (self.tail_len + 1):
            self.tail = self.tail[-self.tail_len - 1:]
            # self.tail.pop(0)


EPISODES = 500
SHOW_EVERY = EPISODES // 3

ACTIONS = 4  # 4 Moves possible
MOVE_DIRECTIONS = 4  # state movement directions
VIEW_LEN = 3
VIEW_AREA = (VIEW_LEN * 2 + 1) ** 2  # Formula
print(f"View area: {VIEW_AREA}")
FIELD_STATES = 3  # 0-Path, 1-Food, 2-Wall/Tail

eps = 0.5
EPS_OFFSET = 0.01
EPS_START_DECAYING = 0
EPS_DECAY_AT = EPISODES // 2
eps_iterator = iter(np.linspace(eps, 0, EPS_DECAY_AT - EPS_START_DECAYING))

size = [MOVE_DIRECTIONS] + [FIELD_STATES * VIEW_AREA] + [ACTIONS]
print(f"Size:\n{size}")

q_table = np.random.uniform(-5, 5, size=size)
# print(q_table.shape)







# with open('run_params.txt', 'at') as file:
#     file.write(f"RUN: {run_num:>3d}, Episodes: {EPISODES:>6d}, Discount: {DISCOUNT:>4.2f}, Learning-rate: {LEARNING_RATE:>4.2f}, "
#                f"Spaces: {STATE_SPACES:>3d}, "
#                f"Eps-init: {eps:>2.4f}, Eps-end: {END_EPS:>2.4f}, Eps-decay-at: {END_EPSILON_DECAYING:>6d}, "
#                f"Timeframe: {TIME_FRAME:>6d}, Eps-toggle: {str(EPS_TOGGLE):>6}")
#     file.write('\n')


direction, state = Game().reset()
for episode in range(1, EPISODES):
    if not episode % SHOW_EVERY:
        render = True
    else:
        render = False

    game = Game(food_ammount=1, render=render, view_len=VIEW_LEN)
    valid = True
    direction, state = game.reset()

    while valid:
        action = np.random.randint(0, 4)

        valid, rewards, state = game.step(action)
        if render:
            game.draw()
            time.sleep(0.01)

    # print(f"Ep[{episode}]: {game.score}")
    break
