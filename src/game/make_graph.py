import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
import sys


name = "Snake1"
TIME_WINDOW = 1000
FRAME = TIME_WINDOW // 2

avg = []
avg_score = []
stats = np.load("last_stats_0.npy", allow_pickle=True).item()


for episode in stats['episode']:
    if episode > FRAME:
        ind = episode - FRAME
    else:
        ind = episode
    values = stats['food_eaten'][ind:episode+FRAME]
    score_vals = stats['score'][ind:episode+FRAME]
    try:
        avg.append(sum(values) / len(values))
        avg_score.append(sum(score_vals) / len(score_vals))
    except ZeroDivisionError:
        avg.append(0)
        avg_score.append(0)


file_path = f"graphs/{name}" + '.png'
if os.path.isfile(file_path):
    print("File exists! Exiting.")
    sys.exit()

style.use('ggplot')
plt.figure(figsize=(16, 9))
plt.subplot(211)

plt.scatter(stats['episode'], stats['food_eaten'], marker='s', alpha=0.1, c='g', label='Food-eaten')

plt.title(f"{name}, alfa=0.1, discount=0.9")
plt.plot(stats['episode'], avg, label='Average food', c='b')
plt.ylabel("Food count")
plt.legend(loc=2)

plt.subplot(212)

plt.plot(stats['episode'], avg_score, label='Average-score', c='b')
plt.scatter(stats['episode'], stats['score'], marker='s', alpha=0.1, c='m', label='score')
plt.ylim([-1000, 1500])
plt.xlabel("Episodes")
plt.ylabel("Points")
plt.legend(loc=2)


os.makedirs("graphs", exist_ok=True)
plt.savefig(file_path)
plt.show()

