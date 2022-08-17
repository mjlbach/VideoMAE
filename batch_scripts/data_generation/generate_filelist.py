import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
#/viscam/data/SomethingSomethingV2/20bn-something-something-v2/
parser.add_argument("--video_folder", required=True)
parser.add_argument("--splits", default=64)
#parser.add_argument("--out_path", required=True)
args = parser.parse_args()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

videos = os.listdir(args.video_folder)
videos = [os.path.join(args.video_folder, video) for video in videos]
labels = range(len(videos))

videos = list(chunks(videos, args.splits))
labels = list(chunks(labels, args.splits))

p = Path('./ssv2').mkdir(parents=True, exist_ok=True)
for i in range(args.splits):
    with open(f"./ssv2/ssv2{i}.txt",'w') as f:
        for i, j in zip(labels[i], videos[i]):
            f.write(f"{j} {i}\n")

