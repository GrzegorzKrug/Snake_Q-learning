import pygame
from pygame.locals import *
import numpy as np
import time

# class Player:
#     def __init__(self):
#         self.x = 10
#         self.y = 10
#         self.speed = 1
#         self.direction = 0
#
#     # def update(self):
#         # if self.direction == 0
#
#     def moveRight(self):
#         self.x += self.speed
#
#     def moveLeft(self):
#         self.x -= self.speed
#
#     def moveUp(self):
#         self.y += self.speed
#
#     def moveDown(self):
#         self.y -= self.speed
#
#
# def deco_init_quit(fun):
#     def wrapper(*args, **kwargs):
#         pygame.init()
#         out = fun(*args, **kwargs)
#         pygame.quit()
#         print('Quit')
#         return out
#     return wrapper
# @deco_init_quit

class Game:
    def __init__(self, width=1.4e3, height=8e2):
        pygame.init()
        width = int(width)
        height = int(height)
        
        self.size = self.width, self.height = width, height
        self.screen = pygame.display.set_mode(self.size)
    
        self.score = 0
        
        self.rect_size = 25
        self.speed_multiplier  = 1
        self.direction = 1
        
        self.x = (width / 2) // self.rect_size * self.rect_size
        self.y = (height / 2) // self.rect_size * self.rect_size

        self.tail_len = 50
        self.tail = [[self.x, self.y]]

        self.food = []
##        
    def __del__(self):
        # print('Deleted')
##        pygame.display.quit()
##        self.scree.quit()
        pygame.quit()
##        exit()

    def play(self):
        f_run = True
        while f_run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    f_run = False
                    break

            # Keyboard Input section
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and self.direction != 1:
                self.direction = 3
            elif keys[pygame.K_RIGHT] and self.direction != 3:
                self.direction = 1
            elif keys[pygame.K_UP] and self.direction != 2:
                self.direction = 0
            elif keys[pygame.K_DOWN] and self.direction != 0:
                self.direction = 2

            # Moving Section
            self.move_snake()
            hit1 = self.check_border()
            hit2 = self.check_collision(self.x, self.y)
            if hit1 or hit2:
                f_run = False  # <<---- Collision, Disable loop
                if hit2:
                    self.move_snake(reverse=True)
                self.speed_multiplier = 0
                color = (255, 0, 0)
            else:
                color = (180, 130, 255)

            self.update_tail()
            self.eat_food()
            
            # Drawing Section --------
            self.screen.fill((30, 30, 50))
            pygame.draw.rect(self.screen, (35,50,50), (self.tail[0][0], self.tail[0][1], self.rect_size, self.rect_size))
            for tail in self.tail[1:]:
                pygame.draw.rect(self.screen, (35,120,50), (tail[0], tail[1], self.rect_size, self.rect_size))
            pygame.draw.rect(self.screen, color, (self.x, self.y, self.rect_size, self.rect_size))
            
            self.food_event(True, self.tail_len)            
            for food in self.food:
                pygame.draw.rect(self.screen, (0, 255, 0), (food[0], food[1], self.rect_size, self.rect_size))

            self.display_score()
            pygame.display.update()              
            time.sleep(.0815)

        time.sleep(3)
        pygame.display.quit()
##        pygame.quit()
        
    def display_score(self):
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render('Score = ' + str(self.score), False, (255, 255, 255))
        self.screen.blit(textsurface,(0,0))
        
    def eat_food(self):
        for i, food in enumerate(self.food):
            if self.check_food_collision(food[0], food[1]):
                self.food.pop(i)
                self.tail_len += 1
                self.score += 1
                break
    
    def food_event(self, multiFood=False, n=10):        
        if multiFood and len(self.food) < n\
           or len(self.food) < 1:
            self.place_food()                            
                
    def place_food(self):
        while True:
            rx, ry = np.random.rand() * (self.width - 1), np.random.rand() * (self.height - 1)
            rx, ry = round(rx // self.rect_size * self.rect_size),\
                     round(ry // self.rect_size * self.rect_size)

            if self.check_food_collision(rx, ry):
                continue
            break    
        self.food += [[rx, ry]]


    def check_food_collision(self, x, y):
        for pos in self.tail:
            if [x, y] == pos:                
                return True
        return False
    
    def check_collision(self, x, y):
        for pos in self.tail[1:-1]:
            if [x, y] == pos:
                # self.speed_multiplier = 0
                return True
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

    def update_tail(self):
        if self.tail[-1] != [self.x, self.y]:
            self.tail += [[self.x, self.y]]

        if len(self.tail) > (self.tail_len + 1):
            self.tail = self.tail[-self.tail_len-1:]
            # self.tail.pop(0)

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
                hit = True
             # if colision with wall return hit
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

G1 = Game()
G1.play()

print('Score = ', G1.score)
input('Bye....')
