VIEW_LEN = 3
VIEW_AREA = VIEW_LEN * 2 + 1

EPOCHS = 2000
SIM_COUNT = 10
full_game = 500
MINIBATCH_SIZE = full_game * SIM_COUNT // 10
REPLAY_MEMORY_SIZE = 5 * full_game * SIM_COUNT  # 10 full games 3k each
MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE

DISCOUNT = 0.9
AGENT_LR = 0.00001
FREE_MOVE = False

MODEL_NAME = f"Model15-Relu-Ag10-View-{VIEW_LEN}-{FREE_MOVE}--1e-3-B_{MINIBATCH_SIZE}-Lr{AGENT_LR}"

ALLOW_TRAIN = True
SAVE_PICS = ALLOW_TRAIN

STATE_OFFSET = 0
FIRST_EPS = 0.5
RAMP_EPS = 0.4
INITIAL_SMALL_EPS = 0.15
END_EPS = 0.0
EPS_INTERVAL = 50

SHOW_EVERY = 100
RENDER_DELAY = 0.01
LOAD_MODEL = True
SHOW_LAST = False
PLOT_ALL_QS = True
PLOT_FIRST_QS = True
COMBINE_QS = True

MOVE_PENALTY = -.1
FOOD_REWARD = 15
FOOD_REWARD_RISING = True
DEAD_PENALTY = -50

SOUND_ALERT = True
