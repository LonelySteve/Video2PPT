import cv2
import numpy as np
import os
import sys
from pathlib import Path
import shutil


def average_hash(img):
    ihash = img > np.mean(img)
    return ihash


def calc_diff(prev, curr, w=50, h=50):
    try:
        prev = cv2.resize(prev, (w, h))
        curr = cv2.resize(curr, (w, h))
        hash_prev = prev > np.mean(prev)
        hash_curr = curr > np.mean(curr)
        diff = hash_prev & hash_curr
        return np.sum(diff)
    except:
        return 0


def proc(fname):
    out_dir = Path(Path(fname).stem)
    if out_dir.exists():
        print("clear output folder")
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    diff = []
    frames = []
    cap = cv2.VideoCapture(fname)

    f = 0
    prev = None
    prev_diff = 1000
    prev_frame = -1000

    os.chdir(out_dir)

    while cap.isOpened():
        f += 1
        ret, frame = cap.read()
        if frame is None:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_diff = calc_diff(prev, curr)
        diff.append(curr_diff)

        if abs(curr_diff - prev_diff) > 50 and f - prev_frame > 10:
            frames.append(frame)
            out_path = f"{f}-{len(frame)}.png"
            print(out_path)
            cv2.imwrite(out_path, frame)
            prev_frame = f

        prev = curr
        prev_diff = curr_diff

    print(f"Task: {Path(fname).stem} finished")

    return frames


proc(sys.argv[1])
