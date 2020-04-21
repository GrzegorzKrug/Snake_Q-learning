VIEW_LEN = 3
VIEW_AREA = VIEW_LEN * 2 + 1

EPOCHS = 100
SIM_COUNT = 1
MINIBATCH_SIZE = SIM_COUNT * 100  # * SIM_COUNT

REPLAY_MEMORY_SIZE = 15 * SIM_COUNT * 1000
MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE

DISCOUNT = 0.95
AGENT_LR = 0.001

MODEL_NAME = f"View-{VIEW_LEN}--1e-3-B_{MINIBATCH_SIZE}"
LOAD_MODEL = True
ALLOW_TRAIN = True
SAVE_PICS = ALLOW_TRAIN

STATE_OFFSET = 0
FIRST_EPS = 0.6
RAMP_EPS = 0.5
INITIAL_SMALL_EPS = 0.15
END_EPS = 0.0
EPS_INTERVAL = EPOCHS // 3

SHOW_EVERY = EPOCHS // 5
TRAIN_EVERY = 1
CLONE_EVERY_TRAIN = 1

SHOW_LAST = False
PLOT_ALL_QS = True
COMBINE_QS = True

