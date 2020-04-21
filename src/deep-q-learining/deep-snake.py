import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import settings
import datetime
import pygame
import random
import keras
import time
import os

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import Adam
from matplotlib import style
from collections import deque


class Game:
    _count = 0

    def __init__(self, width=1.4e3, height=8e2, render=False, food_ammount=3,
                 view_len=4, time_out=3000):
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
        text_surface = my_font.render('Food-eaten = ' + str(self.score), False, (255, 255, 255))
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

    def observation(self):
        """

        :return:
        int:
            direction in range<0, 3>
        list:
            food-head relative position shape=(2)
            positive x - move right
            positive y - move top
        list:
            view-area, shape(2*viewlen + 1), elements: 0-path, 1-collision
        """
        a = self.view_len * 2 + 1
        view_area = np.zeros((a, a), dtype=int)
        for iy in range(a):
            for ix in range(a):
                x = int(self.x + (ix - self.view_len) * self.rect_size)
                y = int(self.y + (iy - self.view_len) * self.rect_size)

                if x > self.width or x < 0 or \
                        y >= self.height or y < 0:
                    view_area[iy, ix] = 1
                    continue

                for tail in self.tail:
                    if [x, y] == tail:
                        view_area[iy, ix] = 1
                        break

                for food in self.food:
                    if [x, y] == food:
                        view_area[iy, ix] = 2
                        break

        if len(self.food) > 0:
            food = self.food[0]
        else:
            food = [0, 0]
            print(f"No Food info")

        food_info = [
                (food[0] - self.x)/self.width,
                (self.y - food[1])/self.height]

        out = food_info + [self.direction / 4]
        state = (view_area, out)
        return state

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
            bool: done
            int: reward
            tuple: observation
                2DList: ViewArea
                Tuple:
                    float: relative food x position
                    float: relative food y position
                    int: direction
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
        reward = self.eat_food() * 10  # Eaten food is worth 5
        if not FREE_MOVE:
            reward -= 1
        state = self.observation()

        if not f_run:  # Dead
            self.done = True
            reward = -50

        if self.current_time >= self.time_out:
            print(f"Timeout! score: {self.score}")
            self.done = True
            reward = -1

        return self.done, reward, state

    def update_tail(self):
        if self.tail[-1] != [self.x, self.y]:
            self.tail += [[self.x, self.y]]

        if len(self.tail) > (self.tail_len + 1):
            self.tail = self.tail[-self.tail_len - 1:]
            # self.tail.pop(0)


class Agent:
    def __init__(self,
                 minibatch_size, pic_size, direction_with_food_array,
                 action_space,
                 learining_rate=0.001):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{MODEL_NAME}--{dt.tm_mon:>02}-{dt.tm_mday:>02}-" \
                            f"-{dt.tm_hour:>02}--{dt.tm_min:>02}-{dt.tm_sec:>02}"

        self.minibatch_size = minibatch_size
        self.pic_size = pic_size
        self.direction_with_food_array = direction_with_food_array
        self.action_space = action_space
        self.learning_rate = learining_rate
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        if LOAD_MODEL:
            load_success = self.load_model()
            if load_success:
                print(f"Loading model: {MODEL_NAME}")
            else:
                print(f"New model: {MODEL_NAME}")
                self.model = self.create_model()
        else:
            self.model = self.create_model()

        # with open(f"{MODEL_NAME}/model_summary.txt", 'w') as file:
        #     self.model.summary(print_fn=lambda x: file.write(x + '\n'))

    def create_model(self):
        input_1 = Input(shape=self.pic_size)

        layer_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_1)
        layer_1 = MaxPooling2D()(layer_1)
        # layer_1 = Conv2D(32, (2, 2), padding='same', activation='relu')(layer_1)
        # layer_1 = MaxPooling2D()(layer_1)
        layer_1 = Flatten()(layer_1)
        layer_1 = Dense(32, activation='relu')(layer_1)

        input_2 = Input(shape=self.direction_with_food_array)
        layer_2_1 = Dense(16, activation='relu')(input_2)

        merged_vector = keras.layers.concatenate([layer_1, layer_2_1], axis=-1)

        # layer_3 = Dense(32, activation='relu')(merged_vector)
        layer_4 = Dropout(0.2)(merged_vector)
        layer_5 = Dense(16, activation='relu')(layer_4)
        output_layer = Dense(self.action_space, activation='linear')(layer_5)

        model = Model(inputs=[input_1, input_2], outputs=output_layer)
        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss='mse',
                      metrics=['accuracy'])

        plot_model(model, f"{MODEL_NAME}/model.png")
        with open(f"{MODEL_NAME}/model_summary.txt", 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

        return model

    def update_memory(self, state):
        self.memory.append(state)

    def save_model(self):
        # self.model.save_weights(f"{MODEL_NAME}/model", overwrite=True)
        self.model.save(f"{MODEL_NAME}/model")

    def load_model(self):
        if LOAD_MODEL and os.path.isfile(f"{MODEL_NAME}/model"):
            self.model = load_model(f"{MODEL_NAME}/model")
            return True
        else:
            return False

    def train(self):
        if len(self.memory) < MINIBATCH_SIZE:
            return None

        train_data = random.sample(self.memory, MINIBATCH_SIZE)
        old_view = []
        old_info = []
        new_view = []
        new_info = []
        rewards = []
        done_list = []
        actions = []
        for old_state, new_state, reward, action, done in train_data:
            old_view.append(old_state[0])
            old_info.append(old_state[1])
            new_view.append(new_state[0])
            new_info.append(new_state[1])
            actions.append(action)
            rewards.append(reward)
            done_list.append(done)

        input1 = np.array(old_view).reshape((-1, VIEW_AREA, VIEW_AREA, 1))
        input2 = np.array(old_info)
        input3 = np.array(new_view).reshape((-1, VIEW_AREA, VIEW_AREA, 1))
        input4 = np.array(new_info)

        old_qs = self.model.predict([input1, input2])
        new_qs = self.model.predict([input3, input4])

        for old_q, new_q, rew, act, done in zip(old_qs, new_qs,
                                                rewards, actions, done_list):
            if done:
                old_q[act] = rew
            else:
                future_best_val = np.max(new_q)
                old_q[act] = rew + DISCOUNT * future_best_val

        self.model.fit([input1, input2], old_qs,
                       verbose=0, shuffle=False, epochs=1)


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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.compat.v1.Session(config=config)

    EPOCHS = settings.EPOCHS
    SIM_COUNT = settings.SIM_COUNT
    MINIBATCH_SIZE = settings.MINIBATCH_SIZE

    REPLAY_MEMORY_SIZE = settings.REPLAY_MEMORY_SIZE
    MIN_REPLAY_MEMORY_SIZE = settings.MIN_REPLAY_MEMORY_SIZE

    DISCOUNT = settings.DISCOUNT
    AGENT_LR = settings.AGENT_LR
    FREE_MOVE = settings.FREE_MOVE

    MODEL_NAME = settings.MODEL_NAME
    LOAD_MODEL = settings.LOAD_MODEL
    ALLOW_TRAIN = settings.ALLOW_TRAIN
    SAVE_PICS = settings.SAVE_PICS

    STATE_OFFSET = settings.STATE_OFFSET
    FIRST_EPS = settings.FIRST_EPS
    RAMP_EPS = settings.RAMP_EPS
    INITIAL_SMALL_EPS = settings.INITIAL_SMALL_EPS
    END_EPS = settings.END_EPS
    EPS_INTERVAL = settings.EPS_INTERVAL

    SHOW_EVERY = settings.SHOW_EVERY

    SHOW_LAST = settings.SHOW_LAST
    PLOT_ALL_QS = settings.PLOT_ALL_QS
    COMBINE_QS = settings.COMBINE_QS

    os.makedirs(MODEL_NAME, exist_ok=True)

    "Environment"
    ACTIONS = 3  # Turn left, right or none
    FIELD_STATES = 2
    VIEW_AREA = settings.VIEW_AREA
    VIEW_LEN = settings.VIEW_LEN

    Predicts = [[], []]

    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "food_eaten": []
    }

    agent = Agent(minibatch_size=MINIBATCH_SIZE,
                  pic_size=(VIEW_AREA, VIEW_AREA, 1,),
                  direction_with_food_array=(3,),
                  action_space=3)

    try:
        episode_offset = np.load(f"{MODEL_NAME}/last-episode-num.npy", allow_pickle=True)
    except FileNotFoundError:
        episode_offset = 0

    eps_iter = iter(np.linspace(RAMP_EPS, END_EPS, EPS_INTERVAL))
    time_start = time.time()
    for episode in range(0, EPOCHS):
        if ALLOW_TRAIN:
            agent.train()
            if not episode % 1000:
                agent.save_model()
                np.save(f"{MODEL_NAME}/last-episode-num.npy", episode + episode_offset)
            if not episode % SHOW_EVERY:
                render = True
            else:
                render = False

            if episode == EPOCHS - 1:
                eps = 0
                render = True
                if SHOW_LAST:
                    input("Last agent is waiting...")
            elif episode == 0:
                eps = 0
                render = True
            elif episode < EPS_INTERVAL / 2:
                eps = FIRST_EPS
            elif episode < EPS_INTERVAL:
                eps = 0.3
            else:
                try:
                    eps = next(eps_iter)
                except StopIteration:
                    eps_iter = iter(np.linspace(INITIAL_SMALL_EPS, END_EPS, EPS_INTERVAL))
                    eps = next(eps_iter)
        else:
            render = True
            eps = 0

        game = None  # Close screen
        game = Game(food_ammount=1, render=render, view_len=VIEW_LEN)
        state = game.reset()
        area, more_info = state
        done = False
        score = 0
        step = 0
        while not done:
            if render and step > 700:
                render = False
                print("Render stopped.")
            step += 1
            old_state = state

            if eps > np.random.random():
                action = np.random.randint(0, ACTIONS)
            else:
                area, more_info = old_state
                area = np.array(area).reshape((-1, VIEW_AREA, VIEW_AREA, 1))
                more_info = np.array(more_info).reshape(-1, 3)

                prediction = agent.model.predict([area, more_info])[0]
                action = np.argmax(prediction)
                if PLOT_ALL_QS:
                    Predicts[0].append(action)
                    Predicts[1].append(prediction[action])

            done, reward, state = game.step(action=action)
            agent.update_memory((old_state, state, reward, action, done))

            if render:
                game.draw()
                time.sleep(0.008)
            score += reward

        stats['episode'].append(episode+episode_offset)
        stats['eps'].append(eps)
        stats['score'].append(score)
        stats['food_eaten'].append(game.score)

        print(f"Ep[{episode+episode_offset:^7} of {EPOCHS+episode_offset}], food_eaten:{game.score:^3}, "
              f"Eps: {eps:>1.3f}, reward: {score:<6.1f}")

    time_end = time.time()
    pygame.quit()
    style.use('ggplot')
    plt.figure(figsize=(20, 11))
    plt.subplot(211)
    plt.title("Food eaten")
    plt.scatter(
            np.array(stats['episode'])+episode_offset,
            stats['food_eaten'],
            alpha=0.13, marker='s', edgecolors='m', label="Food_eaten"
    )
    plt.legend(loc='best')

    plt.subplot(212)
    plt.plot(stats['episode'], stats['score'], label='Score')
    plt.xlabel("Epoch")
    plt.legend(loc='best')
    if SAVE_PICS:
        plt.savefig(f"{MODEL_NAME}/food-{agent.runtime_name}.png")

    plt.figure(figsize=(20, 11))
    samples = []
    colors = []
    for action, q_val in zip(Predicts[0], Predicts[1]):
        color = 'g' if action == 0 else 'm' if action == 1 else 'b'
        samples.append(q_val)
        colors.append(color)

    plt.scatter(range(len(samples)), samples, c=colors, alpha=0.3, s=10, marker='.')
    plt.title("Movement evolution in time:\n"
              "Left Green, Red None, Blue Right")
    plt.xlabel("Sample")
    plt.ylabel("Q-value")
    if SAVE_PICS:
        plt.savefig(f"{MODEL_NAME}/Qs-{agent.runtime_name}.png")

    if ALLOW_TRAIN:
        agent.save_model()
        np.save(f"{MODEL_NAME}/last-episode-num.npy", EPOCHS + episode_offset)

    print(f"Run ended: {MODEL_NAME}")
    print(f"Time elapsed: {(time_end-time_start)/60:3.1f}m, "
          f"{(time_end-time_start)/EPOCHS:3.1f} s per episode")

    if not SAVE_PICS:
        plt.show()

