from deep_snake import Game, Agent

import settings
import numpy as np
import os
import pygame


if __name__ == "__main__":
    done = False
    score = 0
    step = 0

    path = f"{settings.MODEL_NAME}/game28_V-4_2000"
    if os.path.isdir(path):
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.removedirs(path)
    os.makedirs(path, exist_ok=True)

    game = Game(food_ammount=settings.FOOD_COUNT, render=True, view_len=settings.VIEW_LEN, free_moves=settings.TIMEOUT)
    observation = game.reset()

    ACTIONS = 4
    INPUT_SHAPE = (settings.VIEW_AREA * settings.VIEW_AREA + 2,)
    agent = Agent(input_shape=INPUT_SHAPE,
                  action_space=ACTIONS)

    while not done:
        game.draw()
        surface = pygame.display.get_surface()
        pygame.image.save(surface, path + f"/{step}.png")

        predictions = agent.model.predict(observation.reshape(-1, *INPUT_SHAPE))
        action = np.argmax(predictions)
        observation, reward, done = game.step(action=action)
        step += 1

    game.draw()
    surface = pygame.display.get_surface()
    pygame.image.save(surface, path + f"/{step}.png")
