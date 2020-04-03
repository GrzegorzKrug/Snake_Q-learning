import pygame
# from pygame.locals import *
import numpy as np
import time


class SquareMovement:
    def __init__(self, dist=15, timeout=30):
        self._now = -1
        self.now = -1
        self.dist = dist
        self.timeout = timeout

    def __next__(self):
        self._now += 1
        if not self._now % self.dist:
            self.now = (self.now + 1) % 4

        if self._now > self.timeout:
            return 0
        return self.now

    def __repr__(self):
        return f"Square: {self.now}"


class Game:
    def __init__(self, width=1.4e3, height=8e2, render=True):
        if render:
            pygame.init()
            width = int(width)
            height = int(height)
            self.screen = pygame.display.set_mode((width, height))

        self.render = render
        self.food_on_screen = 150

        self.size = self.width, self.height = width, height

        self.square_move = SquareMovement()
        self.move_time = 1
        self.score = 0

        self.rect_size = 25
        self.speed_multiplier = 1
        self.direction = 1

        self.x = (width / 2) // self.rect_size * self.rect_size
        self.y = (height / 2) // self.rect_size * self.rect_size

        self.tail_len = 10
        self.tail = [[self.x, self.y]]
        self.done = False
        self.food = []
        self.food_refill(True, self.food_on_screen)

    def __del__(self):
        pygame.quit()

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
        while multi_food and len(self.food) < n\
           or len(self.food) < 1:
            self.place_food()

    def observation_area(self, area=7):
        direction = self.direction
        area = np.zeros_like(7, 7)

        return (direction, area)

    def observation_lines(self):
        self.x
        self.y
        pass

    def move_snake(self, reverse=False):
        # Directions
        #   0       Up
        # 3   1     Left / Right
        #   2       Down
        current_speed = self.speed_multiplier * self.rect_size
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
    
    def place_food(self):
        while True:
            rx, ry = np.random.rand() * (self.width - 1), np.random.rand() * (self.height - 1)
            rx, ry = round(rx // self.rect_size * self.rect_size),\
                     round(ry // self.rect_size * self.rect_size)

            if self.check_collision_with_food(rx, ry):
                continue
            break
        self.food += [[rx, ry]]

    def play(self, delay=0.03):
        valid = True
        _color = (130, 255, 255)
        index = 0
        try:
            while valid:

                if self.render:
                    self.screen.fill((30, 30, 50))
                    pygame.draw.rect(self.screen, (50, 150, 130),
                                     (self.tail[0][0], self.tail[0][1], self.rect_size, self.rect_size))  # last tail piece
                    for tail in self.tail[1:]:
                        pygame.draw.rect(self.screen, (35, 120, 50), (tail[0], tail[1], self.rect_size, self.rect_size))
                    pygame.draw.rect(self.screen, _color, (self.x, self.y, self.rect_size, self.rect_size))

                    for food in self.food:
                        pygame.draw.rect(self.screen, (0, 255, 0), (food[0], food[1], self.rect_size, self.rect_size))

                    self.display_score()
                    pygame.display.update()

                    # surf = pygame.display.get_surface()
                    # pygame.image.save(surf, f'image{index}.png')

                    time.sleep(delay)

                valid, reward, state = self.step()
                index += 1
        finally:
            if self.render:
                pygame.display.quit()

    def step(self):
        if self.done:
            print("Run has ended")
        f_run = True
        render = self.render
        self.food_refill(True, self.food_on_screen)

        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    f_run = False
                    break

        self.direction = next(self.square_move)
        self.move_snake()
        hit1 = self.check_border()
        hit2 = self.check_collision_with_obstacles(self.x, self.y)
        if hit1 or hit2:
            f_run = False  # <<---- Collision, Disable loop
            if hit2:
                self.move_snake(reverse=True)
            self.speed_multiplier = 0
            _color = (255, 0, 0)
        else:
            _color = (130, 255, 255)

        self.update_tail()
        reward = self.eat_food()
        obs = self.observation_area()
        # Drawing Section --------
        if not f_run:
            self.done = True
            reward = -10
        return f_run, reward, obs

    def update_tail(self):
        if self.tail[-1] != [self.x, self.y]:
            self.tail += [[self.x, self.y]]

        if len(self.tail) > (self.tail_len + 1):
            self.tail = self.tail[-self.tail_len-1:]
            # self.tail.pop(0)

    # def player_input(self):
    #     if self.render:
    #         time_0 = time.time()
    #         # while time.time() - time0 < self.move_time:  # TIME FRAME FOR INPUT
    #             # Keyboard Input section
    #         keys = pygame.key.get_pressed()
    #         # print(keys)
    #         if keys[pygame.K_LEFT] and self.direction != 1:
    #             self.direction = 3
    #             print("Selected 3")
    #             # break
    #         elif keys[pygame.K_RIGHT] and self.direction != 3:
    #             self.direction = 1
    #             print("Selected 1")
    #             # break
    #         elif keys[pygame.K_UP] and self.direction != 2:
    #             self.direction = 0
    #             print("Selected 0")
    #             # break
    #         elif keys[pygame.K_DOWN] and self.direction != 0:
    #             self.direction = 2
    #             print("Selected 2")
    #             # break
    #         else:
    #             print("No key selected")
    #         # if time.time() - time_0 > self.move_time:
    #             # break
    #         time.sleep(0.001)
    #         time.sleep(0.05)


# G1 = Game(render=False)
# G1.play()

G2 = Game()
G2.play()

# print('Score1 = ', G1.score)
print('Score2 = ', G2.score)
# time.sleep(5)
# input('Bye....')
