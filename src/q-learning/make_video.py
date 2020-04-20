import cv2
import glob


RUN_NUM = 20
DIRECTORY = "last_qtable"

files = glob.glob(f"{DIRECTORY}/*png", recursive=True)
last_file = len(files)


def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'Snake-{DIRECTORY}.mp4', fourcc, 30.0, (1400, 800))

    for i in range(last_file):
        img_path = f"{DIRECTORY}/image_{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


if __name__ == "__main__":
    make_video()

