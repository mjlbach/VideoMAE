import pandas as pd
import os

output_path = "/vision/group/ego4d/v1/full_scale"
mp4_list = [os.path.join(output_path, item) for item in os.listdir(output_path) if item.endswith('.mp4')]
labels = ["" + str(i) for i in range(len(mp4_list))]
df = pd.DataFrame({"mp4": mp4_list, "labels": labels})
df.to_csv("/svl/u/mjlbach/train.csv", header=False, index=False, sep=' ')
