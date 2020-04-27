VIEW_LEN = 10
VIEW_AREA = VIEW_LEN * 2 + 1

SIM_COUNT = 10
EPOCHS = 1000
TIMEOUT = 200
TRAIN_MAX_MIN_DURATION = 10
full_game = 2000
FOOD_COUNT = 1

MINIBATCH_SIZE = 2000
MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE

DISCOUNT = 0.9
AGENT_LR = 1e-5
ALLOW_TRAIN = True
TRAIN_ALL = False

SHOW_EVERY = 250
RENDER_DELAY = 0.03

if TRAIN_ALL:
    REPLAY_MEMORY_SIZE = MINIBATCH_SIZE
else:
    REPLAY_MEMORY_SIZE = 5 * full_game * SIM_COUNT  # 10 full games 3k each

MODEL_NAME = f"Model27-NewGame-View-{VIEW_LEN}--B_{MINIBATCH_SIZE}"

# Training params
STATE_OFFSET = 0
FIRST_EPS = 0.5
RAMP_EPS = 0.4
INITIAL_SMALL_EPS = 0.3
END_EPS = 0.0
EPS_INTERVAL = 100

# Reward
MOVE_PENALTY = -0.1
FOOD_REWARD = 50
FOOD_REWARD_RISING = False
DEATH_PENALTY = -50

# Settings
SAVE_PICS = ALLOW_TRAIN
LOAD_MODEL = True
FREE_MOVE = False
SHOW_LAST = False
PLOT_ALL_QS = True
PLOT_FIRST_QS = False
COMBINE_QS = True
SOUND_ALERT = True


