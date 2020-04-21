VIEW_LEN = 4
VIEW_AREA = VIEW_LEN * 2 + 1

EPOCHS = 500
SIM_COUNT = 1
MINIBATCH_SIZE = 1000  # * SIM_COUNT

REPLAY_MEMORY_SIZE = SIM_COUNT * 5_000
MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE

DISCOUNT = 0.95
AGENT_LR = 0.001
FREE_MOVE = False

MODEL_NAME = f"Model5-FoodinArea-1xCov5-View-{VIEW_LEN}--FreeMove-{FREE_MOVE}--1e-3-B_{MINIBATCH_SIZE}"
LOAD_MODEL = True
ALLOW_TRAIN = True
SAVE_PICS = ALLOW_TRAIN

STATE_OFFSET = 0
FIRST_EPS = 0.6
RAMP_EPS = 0.5
INITIAL_SMALL_EPS = 0.15
END_EPS = 0.0
EPS_INTERVAL = 50

SHOW_EVERY = 100

SHOW_LAST = False
PLOT_ALL_QS = True
COMBINE_QS = True

