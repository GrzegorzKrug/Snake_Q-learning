VIEW_LEN = 4
VIEW_AREA = VIEW_LEN * 2 + 1

FOOD_COUNT = 1
SIM_COUNT = 10
EPOCHS = 10000
TIMEOUT = 200
TRAIN_MAX_MIN_DURATION = 20

DISCOUNT = 0.9
AGENT_LR = 1e-3
ALLOW_TRAIN = True

SHOW_EVERY = 500
RENDER_DELAY = 0.02

REPLAY_MEMORY_SIZE = 10000
MIN_BATCH_SIZE = 200
MAX_BATCH_SIZE = 2000

MODEL_NAME = f"StepModel-30-StepTraining--View-{VIEW_LEN}--MB_{MIN_BATCH_SIZE}"

# Training params
STATE_OFFSET = 0
FIRST_EPS = 0.5
RAMP_EPS = 0.4
INITIAL_SMALL_EPS = 0.3
END_EPS = 0.005
EPS_INTERVAL = 50

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


