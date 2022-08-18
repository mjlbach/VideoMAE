import os
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video_folder", required=True)
parser.add_argument("--split", nargs='+', help='Train-test-val split', type=float, default=[0.8, 0.1, 0.1])
args = parser.parse_args()
#
# def absolute_file_paths(directory):
#     breakpoint()
#     path = os.path.abspath(directory)
#     return [entry.path for entry in os.scandir(path) if entry.is_file()]

# "/vision/group/ego4d/v1/full_scale"
mp4_list = list(Path(args.video_folder).rglob("*.mp4"))
labels = ["" + str(i) for i in range(len(mp4_list))]

assert np.sum(args.split) == 1
train, validate, test = np.split(mp4_list, [int(len(mp4_list)*args.split[0]), int(len(mp4_list)*(args.split[0] + args.split[1]))])

with open("./train.csv",'w') as f:
    for i in train:
        f.write(f"{i} 0\n")

with open("./val.csv",'w') as f:
    for i in validate:
        f.write(f"{i} 0\n")

with open("./test.csv",'w') as f:
    for i in test:
        f.write(f"{i} 0\n")
