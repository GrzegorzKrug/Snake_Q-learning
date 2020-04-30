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
from keras.models import Model, load_model, Sequential
from keras.utils import plot_model
from keras.optimizers import Adam
from collections import deque
from matplotlib import style
from keras import backend


class Game:
    _count = 0

    def __init__(self, food_ammount=1, snake_len=3, width=30, height=30, free_moves=200, view_len=3,
                 random_start=True, render=False):
        """"""
        self.render = render
        self.food_limit = food_ammount
        self.view_len = view_len
        self.area_len = self.view_len * 2 + 1
        self.free_moves = free_moves
        self.initial_snake_len = snake_len
        self.random_start = random_start
        self.MOVE_PENALTY = settings.MOVE_PENALTY
        self.FOOD_REWARD = settings.FOOD_REWARD
        self.DEATH_PENALTY = settings.DEATH_PENALTY

        self.width, self.height = width, height
        self.box_size = 25
        self.size = self.width * self.box_size, self.height * self.box_size

        self.score = 0
        self.direction = 0
        self.moves_left = self.free_moves
        self.time = 0
        self.done = False

        self.head = [0, 0]
        self.tail = deque([self.head] * self.initial_snake_len, maxlen=self.width*self.height+1)
        self.snake_len = self.initial_snake_len
        self.Food = []
        self.fill_food()

        self._reset()
        Game._count += 1

        if render:
            pygame.init()
            width = int(width)
            height = int(height)
            self.screen = pygame.display.set_mode((self.box_size * width, self.box_size * height))

    def __del__(self):
        Game._count -= 1
        if Game._count < 1:
            pygame.quit()

    def _reset(self):
        self.score = 0
        self.direction = 0
        self.moves_left = self.free_moves
        self.time = 0
        self.done = False

        if self.random_start:
            self.head = [np.random.randint(1, self.width-1), np.random.randint(1, self.height-1)]
        else:
            self.head = [self.width // 2, self.height//2]

        self.tail = deque([self.head] * self.initial_snake_len, maxlen=self.width * self.height + 1)
        self.snake_len = self.initial_snake_len
        self.Food = []
        self.fill_food()

    def reset(self):
        self._reset()
        obs = self.observation()
        return obs

    def fill_food(self):
        while len(self.Food) < self.food_limit:
            self.place_food()

    def place_food(self):
        while True:
            valid_to_brake = True
            new_food_pos = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            if new_food_pos == self.head:
                # Repeat while with new food
                continue
            for food in self.Food:
                if food == new_food_pos:
                    valid_to_brake = False
                    break
            if valid_to_brake: break
        self.Food.append(new_food_pos)

    def observation(self):
        area = np.ones((self.area_len, self.area_len))
        for arr_y in range(self.area_len):
            for arr_x in range(self.area_len):
                y = self.head[1] - (arr_y - self.view_len)
                x = self.head[0] - self.view_len + arr_x
                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    area[arr_y, arr_x] = 2
                    continue

                skip_tail = False
                for food in self.Food:
                    if [x, y] == food:
                        area[arr_y, arr_x] = 0
                        skip_tail = True
                        break
                if skip_tail:
                    continue
                for tail in self.tail:
                    if [x, y] == tail:
                        area[arr_y, arr_x] = 2

        area = (area / 2).ravel()
        food_relative_pos = (np.array(self.head) - np.array(self.Food[0])) / np.max(self.size)
        output = np.concatenate([area, food_relative_pos])
        return output

    def random_action(self):
        """ Return valid action in current situation"""
        new_direction = (self.direction + np.random.randint(-1, 2)) % 4
        return new_direction

    def move_snake(self, new_direction):
        """
        Move snake, and update environment
        Directions
          0   Up
        3   1 Left / Right
          2   Down
        Parameters
        ----------
        new_direction - int <0, 3>

        Returns
        -------
        collision - boolean
        food_eaten - boolean
        action_valid - boolean
        """
        # Check if action is valid
        if self.direction == 0 and new_direction == 2 or \
                self.direction == 1 and new_direction == 3 or \
                self.direction == 2 and new_direction == 0 or \
                self.direction == 3 and new_direction == 1:
            action_valid = False
        else:
            action_valid = True

        if action_valid:
            new_direction = new_direction
        else:
            new_direction = self.direction

        new_x = self.head[0] + (1 if new_direction == 1 else -1 if new_direction == 3 else 0)
        new_y = self.head[1] + (1 if new_direction == 2 else -1 if new_direction == 0 else 0)
        new_pos = [new_x, new_y]
        if new_x >= self.width or new_x < 0 or new_y >= self.width or new_y < 0:
            collision = True
        else:
            collision = False

        if not collision:
            for tail in list(self.tail)[1:]:  # First tail segment will move
                if new_pos == tail:
                    collision = True
                    break

        food_eaten = False
        if not collision:
            self.head = new_pos
            for f_index, food in enumerate(self.Food):
                if self.head == food:
                    self.Food.pop(f_index)
                    self.snake_len += 1
                    food_eaten = True
                    break

            self.update_tail()
        self.direction = new_direction
        return collision, food_eaten, action_valid

    def update_tail(self):
        self.tail.append(self.head)
        while len(self.tail) > self.snake_len:  # Head is separate
            self.tail.popleft()

    def step(self, action=None):
        """

        Parameters
        ----------
        action

        Returns
        -------
        observation
        reward
        done

        """
        self.moves_left -= 1
        self.time += 1
        if self.done:
            print("Game has been already ended!")
            done = True
        else:
            done = False

        collision, food, action_valid = self.move_snake(action)

        if collision:
            done = True
        elif self.moves_left < 1 and not food:
            done = True

        if not action_valid:
            reward = self.DEATH_PENALTY * 2
        elif collision:
            reward = self.DEATH_PENALTY
        elif self.moves_left < 1:
            reward = self.MOVE_PENALTY * 2
        elif food:
            reward = self.FOOD_REWARD
        else:
            reward = self.MOVE_PENALTY

        if food:
            self.moves_left = self.free_moves
            self.score += 1

        if done:
            self.done = done
        self.fill_food()
        observation = self.observation()
        return observation, reward, done

    def draw(self):
        if self.done:
            head_color = (255, 0, 0)
        else:
            head_color = (130, 255, 255)

        self.screen.fill((25, 20, 30))

        # self.screen.fill((40, 40, 45))

        pygame.draw.rect(self.screen, (40, 60, 45),
                         ((self.head[0] - self.view_len) * self.box_size, (self.head[1] - self.view_len) * self.box_size, self.area_len * self.box_size, self.area_len * self.box_size))

        for tail in self.tail:
            pygame.draw.rect(self.screen, (35, 120, 50),
                             (tail[0] * self.box_size, tail[1] * self.box_size, self.box_size,
                              self.box_size))
        pygame.draw.rect(self.screen, head_color,
                         (self.head[0] * self.box_size, self.head[1] * self.box_size, self.box_size,
                          self.box_size))

        for food in self.Food:
            pygame.draw.rect(self.screen, (0, 255, 0), (food[0] * self.box_size, food[1] * self.box_size, self.box_size, self.box_size))

        self.display_score()
        pygame.display.update()

    def display_score(self):
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render('Food-eaten = ' + str(self.score), False, (255, 255, 255))
        self.screen.blit(text_surface, (0, 0))

        text_surface = my_font.render('Moves left = ' + str(self.moves_left), False, (255, 255, 255))
        self.screen.blit(text_surface, (0, 20))


class Agent:
    def __init__(self,
                 input_shape,
                 action_space,
                 min_batch_size=1000,
                 max_batch_size=1000,
                 learining_rate=0.0001,
                 memory_size=10000):

        dt = datetime.datetime.timetuple(datetime.datetime.now())
        self.runtime_name = f"{dt.tm_mon:>02}-{dt.tm_mday:>02}--" \
                            f"{dt.tm_hour:>02}-{dt.tm_min:>02}-{dt.tm_sec:>02}"

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.action_space = action_space
        self.learning_rate = learining_rate
        self.memory = deque(maxlen=memory_size)
        load_success = self.load_model()
        if load_success:
            print(f"Loading model: {MODEL_NAME}")
        else:
            print(f"New model: {MODEL_NAME}")
            self.model = self.create_model()

        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss='mse',
                           metrics=['accuracy'])
        backend.set_value(self.model.optimizer.lr, self.learning_rate)
        self.model.summary()

    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=self.input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))

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
        if os.path.isfile(f"{MODEL_NAME}/model"):
            self.model = load_model(f"{MODEL_NAME}/model")
            return True
        else:
            return False

    def train(self):
        if len(self.memory) < self.min_batch_size:
            return None
        elif len(self.memory) > self.max_batch_size:
            train_data = random.sample(self.memory, self.max_batch_size)
            print(f"Too much data, selecting from: {len(self.memory)} samples")
        else:
            train_data = list(self.memory)

        self.memory.clear()

        Old_states = []
        New_states = []
        Rewards = []
        Dones = []
        Actions = []

        for old_state, new_state, reward, action, done in train_data:
            Old_states.append(old_state)
            New_states.append(new_state)
            Actions.append(action)
            Rewards.append(reward)
            Dones.append(done)

        Old_states = np.array(Old_states)
        New_states = np.array(New_states)
        old_qs = self.model.predict(Old_states)
        new_qs = self.model.predict(New_states)

        for old_q, new_q, rew, act, done in zip(old_qs, new_qs, Rewards, Actions, Dones):
            if done:
                old_q[act] = rew
            else:
                future_best_val = np.max(new_q)
                old_q[act] = rew + DISCOUNT * future_best_val

        self.model.fit(Old_states, old_qs,
                       verbose=0, shuffle=False, epochs=1)


EPOCHS = settings.EPOCHS
SIM_COUNT = settings.SIM_COUNT

REPLAY_MEMORY_SIZE = settings.REPLAY_MEMORY_SIZE
MIN_BATCH_SIZE = settings.MIN_BATCH_SIZE
MAX_BATCH_SIZE = settings.MAX_BATCH_SIZE

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
RENDER_DELAY = settings.RENDER_DELAY

SHOW_LAST = settings.SHOW_LAST
PLOT_ALL_QS = settings.PLOT_ALL_QS
COMBINE_QS = settings.COMBINE_QS

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.compat.v1.Session(config=config)

    os.makedirs(MODEL_NAME, exist_ok=True)

    "Environment"
    ACTIONS = 4  # Turn left, right or none
    VIEW_AREA = settings.VIEW_AREA
    VIEW_LEN = settings.VIEW_LEN
    INPUT_SHAPE = (VIEW_AREA * VIEW_AREA + 2, )  # 2 is more infos
    Predicts = [[], [], [], []]
    Pred_sep = []

    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "food_eaten": [],
            "moves": []
    }

    agent = Agent(min_batch_size=MIN_BATCH_SIZE,
                  max_batch_size=MAX_BATCH_SIZE,
                  input_shape=INPUT_SHAPE,
                  action_space=ACTIONS,
                  memory_size=REPLAY_MEMORY_SIZE,
                  learining_rate=AGENT_LR)

    try:
        episode_offset = np.load(f"{MODEL_NAME}/last-episode-num.npy", allow_pickle=True)
    except FileNotFoundError:
        episode_offset = 0

    eps_iter = iter(np.linspace(RAMP_EPS, END_EPS, EPS_INTERVAL))
    time_start = time.time()
    emergency_break = False

    for episode in range(0, EPOCHS):
        Pred_sep.append(len(Predicts[0]))

        if not episode % SHOW_EVERY:
            render = True
        else:
            render = False

        if episode == EPOCHS - 1 or emergency_break:
            eps = 0
            render = True
            if SHOW_LAST:
                input("Last agent is waiting...")
        elif episode == 0 or not ALLOW_TRAIN:
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

        Games = []  # Close screen
        States = []
        for loop_ind in range(SIM_COUNT):
            if loop_ind == 0:
                game = Game(food_ammount=settings.FOOD_COUNT, render=render, view_len=VIEW_LEN, free_moves=settings.TIMEOUT)
            else:
                game = Game(food_ammount=settings.FOOD_COUNT, render=False, view_len=VIEW_LEN, free_moves=settings.TIMEOUT)
            state = game.reset()
            Games.append(game)
            States.append(state)

        Dones = [False] * len(Games)
        Scores = [0] * len(Games)
        step = 0
        All_score = []
        All_steps = []
        while len(Games):
            if ALLOW_TRAIN:
                agent.train()
                if not episode % 100 and episode > 0:
                    agent.save_model()
                    np.save(f"{MODEL_NAME}/last-episode-num.npy", episode + episode_offset)

            step += 1
            Old_states = np.array(States)

            if eps > np.random.random():
                Actions = [game.random_action() for game in Games]
            else:
                Predictions = agent.model.predict(Old_states)
                Actions = np.argmax(Predictions, axis=1)
                if settings.PLOT_FIRST_QS:
                    Predicts[0].append(Actions[0])
                    Predicts[1].append(Predictions[0][Actions[0]])
                elif PLOT_ALL_QS:
                    for predict in Predictions:
                        Predicts[0].append(predict[0])
                        Predicts[1].append(predict[1])
                        Predicts[2].append(predict[2])
                        Predicts[3].append(predict[3])

            States = []
            assert len(Games) == len(Dones)
            for g_index, game in enumerate(Games):
                state, reward, done = game.step(action=Actions[g_index])
                agent.update_memory((Old_states[g_index], state, reward, Actions[g_index], done))
                Scores[g_index] += reward
                Dones[g_index] = done
                States.append(state)

            if render:
                Games[0].draw()
                time.sleep(RENDER_DELAY)

            for ind_d in range(len(Games) - 1, -1, -1):
                if Dones[ind_d]:
                    if ind_d == 0 and render:
                        render = False
                        pygame.quit()

                    All_score.append(Scores[ind_d])
                    All_steps.append(step)

                    stats['episode'].append(episode+episode_offset)
                    stats['eps'].append(eps)
                    stats['score'].append(Scores[ind_d])
                    stats['food_eaten'].append(Games[ind_d].score)
                    stats['moves'].append(step)

                    Scores.pop(ind_d)
                    Games.pop(ind_d)
                    States.pop(ind_d)
                    Dones.pop(ind_d)

        print(f"Ep[{episode+episode_offset:^7} of {EPOCHS+episode_offset}], "
              f"Eps: {eps:>1.3f} "
              f"avg-score: {np.mean(All_score):^8.1f}, "
              f"avg-steps: {np.mean(All_steps):^7.1f}"
              )
        time_end = time.time()
        if emergency_break:
            break
        elif settings.TRAIN_MAX_MIN_DURATION and (time_end - time_start) / 60 > settings.TRAIN_MAX_MIN_DURATION:
            emergency_break = True

    print("Plotting data now...")
    pygame.quit()
    style.use('ggplot')
    plt.figure(figsize=(20, 11))
    plt.subplot(412)
    plt.suptitle(f"{MODEL_NAME}\nStats")
    plt.scatter(
            np.array(stats['episode']),
            stats['food_eaten'],
            alpha=0.2, marker='s', edgecolors='m', label="Food_eaten"
    )
    plt.legend(loc=2)

    plt.subplot(413)
    plt.scatter(stats['episode'], stats['moves'], label='Moves', color='b', marker='.', s=10, alpha=0.5)
    plt.legend(loc=2)

    plt.subplot(414)
    plt.scatter(stats['episode'], stats['eps'], label='Epsilon', color='k', marker='.', s=10, alpha=1)
    plt.legend(loc=2)

    plt.subplot(411)
    effectiveness = [food / moves for food, moves in zip(stats['food_eaten'], stats['moves'])]
    plt.scatter(stats['episode'], effectiveness, label='Effectiveness', color='g', marker='s', s=10, alpha=0.5)
    plt.xlabel("Epoch")
    plt.subplots_adjust(hspace=0.3)
    plt.legend(loc=2)

    if SAVE_PICS:
        plt.savefig(f"{MODEL_NAME}/food-{agent.runtime_name}.png")

    # BIG PLOT
    plt.figure(figsize=(20, 11))
    samples = []
    colors = []
    plt.scatter(range(len(Predicts[0])), Predicts[0], c='r', label='up', alpha=0.2, s=3, marker='o')
    plt.scatter(range(len(Predicts[1])), Predicts[1], c='g', label='right', alpha=0.2, s=3, marker='o')
    plt.scatter(range(len(Predicts[2])), Predicts[2], c='m', label='down', alpha=0.2, s=3, marker='o')
    plt.scatter(range(len(Predicts[3])), Predicts[3], c='b', label='left', alpha=0.2, s=3, marker='o')
    y_min, y_max = np.min(Predicts), np.max(Predicts)

    for sep in Pred_sep:
        last_line, = plt.plot([sep, sep], [y_min, y_max], c='k', linewidth=0.3, alpha=0.2)

    plt.title(f"{MODEL_NAME}\nMovement 'directions' evolution in time, learning-rate:{AGENT_LR}\n")
    last_line.set_label("Epoch separator")
    plt.xlabel("Sample")
    plt.ylabel("Q-value")
    plt.legend(loc='best')

    if SAVE_PICS:
        plt.savefig(f"{MODEL_NAME}/Qs-{agent.runtime_name}.png")

    if ALLOW_TRAIN:
        agent.save_model()
        np.save(f"{MODEL_NAME}/last-episode-num.npy", episode + 1 + episode_offset)

    print(f"Run ended: {MODEL_NAME}")
    print(f"Time elapsed: {(time_end-time_start)/60:3.1f}m, "
          f"{(time_end-time_start)/EPOCHS:3.1f} s per episode")

    if not SAVE_PICS:
        plt.show()
    # train - clear memory
    if settings.SOUND_ALERT:
        os.system("play -nq -t alsa synth 0.2 sine 150")

