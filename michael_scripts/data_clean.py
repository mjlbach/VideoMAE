import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import decord
import ffmpeg

"""
PREFILE=0
tail -n1 sta_web_weak_to_clean${PREFILE}_320p.txt | cut -c 24- | xargs -i grep -n {} sta_web_weak_to_clean${PREFILE}.txt | cut -c -6
"""


def data_clean(list_file, start_idx=0):
    # prepare
    video_ext = 'mp4'
    new_list_file = list_file.replace('.txt', '_320p.txt')

    # load video file list
    if not os.path.exists(list_file):
        raise RuntimeError(f'list file {list_file} not exist!')

    clips = []
    with open(list_file) as split_f:
        lines = split_f.readlines()
        for line in lines:
            line_info = line.split(' ')
            if len(line_info) < 2:
                raise RuntimeError(f'line info {line_info} missing element.')
            clip_path = os.path.join(line_info[0])
            target = int(line_info[-1])
            clips.append((clip_path, target, line))
    num_videos = len(clips)
    print(f'load list file with {num_videos} videos successfully.')

    directory, _, _ = clips[0]
    outpath = Path(directory).parent
    outpath = outpath.parent / (outpath.name + '_sta_web_140w_320p')

    # process video & save
    for index in range(start_idx, num_videos):
        start_time = time.time()
        print(f'processing video {index + 1} / {num_videos}')

        directory, target, ori_line = clips[index]
        if '.' in directory.split('/')[-1]:
            video_name = directory
        else:
            video_name = '{}.{}'.format(directory, video_ext)

        try:
            # try load video
            decord_vr = decord.VideoReader(video_name, num_threads=1)

            duration = len(decord_vr)
            if duration < 30:
                continue
            video_data = decord_vr.get_batch(list(range(duration))).asnumpy()

            # get the new size (short side size 320p)
            _, img_h, img_w, _ = video_data.shape
            new_short_size = 320
            ratio = float(img_h) / float(img_w)
            if ratio >= 1.0:
                new_w = int(new_short_size)
                new_h = int(new_w * ratio / 2) * 2
            else:
                new_h = int(new_short_size)
                new_w = int(new_h / ratio / 2) * 2
            new_size = (new_w, new_h)

        except Exception as e:
            # skip corrupted video files
            print("Failed to load video from {} with error {}".format(
                video_name, e))
            continue

        # process the video
        outpath.mkdir(parents=True, exist_ok=True)
        new_video_name = outpath.joinpath(Path(video_name).stem + '.mp4')
        new_video_name = str(new_video_name)
        with open(video_name, "rb") as f:
            video_bytes = f.read()
            # video_bytes = client.get(video_name)

        # resize
        proc1 = (ffmpeg.input('pipe:').filter(
            'scale', new_size[0],
            new_size[1]).output(new_video_name).overwrite_output())
        p = subprocess.Popen(
            ['ffmpeg'] + proc1.get_args() +
            ['-hide_banner', '-loglevel', 'quiet', '-nostats'],
            stdin=subprocess.PIPE)

        p.communicate(input=video_bytes)

        new_line = ori_line.replace(video_name, new_video_name)

        with open(new_list_file, 'a+') as new_list_f:
            new_list_f.write(new_line)

        end_time = time.time()
        dur_time = end_time - start_time
        print(f'total time {dur_time} & save video in {new_video_name}')


if __name__ == '__main__':
    list_file = sys.argv[-1]
    n_tasks = 64
    new_start_idxs = [0] * n_tasks
    # data_clean(list_file + str(0) + '.txt')

    with Pool(n_tasks) as p:
        p.starmap(data_clean,
                  [(list_file + str(idx) + '.txt', new_start_idxs[idx])
                   for idx in range(n_tasks)])
