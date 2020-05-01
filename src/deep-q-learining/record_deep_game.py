from deep_snake import Game, Agent

import settings
import numpy as np
import os
import pygame


if __name__ == "__main__":
    done = False
    score = 0
    step = 0
    game_name = "game1"
    path = f"{settings.MODEL_NAME}/{game_name}"
    num = 1
    while os.path.isdir(path):
        num += 1
        game_name = f"game{num}"
        path = f"{settings.MODEL_NAME}/{game_name}"

    os.makedirs(path)

    game = Game(food_ammount=settings.FOOD_COUNT, render=True, view_len=settings.VIEW_LEN, free_moves=settings.TIMEOUT)
    observation = game.reset()

    ACTIONS = 4
    if settings.DUAL_INPUT:
        INPUT_SHAPE = [(settings.VIEW_AREA * settings.VIEW_AREA,), (2,)]  # 2 is more infos
    else:
        INPUT_SHAPE = (settings.VIEW_AREA * settings.VIEW_AREA + 2,)  # 2 is direciton

    agent = Agent(min_batch_size=settings.MIN_BATCH_SIZE,
                  max_batch_size=settings.MAX_BATCH_SIZE,
                  input_shape=INPUT_SHAPE,
                  action_space=ACTIONS,
                  memory_size=settings.REPLAY_MEMORY_SIZE,
                  learining_rate=settings.AGENT_LR,
                  dual_input=settings.DUAL_INPUT)

    while not done:
        game.draw()
        surface = pygame.display.get_surface()
        pygame.image.save(surface, path + f"/{step}.png")
        if settings.DUAL_INPUT:
            area = observation[0].reshape(-1, *INPUT_SHAPE[0])
            direction = observation[1].reshape(-1, 2)
            predictions = agent.model.predict([area, direction])
        else:
            predictions = agent.model.predict(observation.reshape(-1, *INPUT_SHAPE))
        action = np.argmax(predictions)
        observation, reward, done = game.step(action=action)
        step += 1

    game.draw()
    surface = pygame.display.get_surface()
    pygame.image.save(surface, path + f"/{step}.png")
