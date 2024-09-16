import glob
import os
import random
from PIL import Image

PROCESSED_DIR = 'data/processed'
file_paths = glob.glob(os.path.join(PROCESSED_DIR,'**', '*.*'))

with open('data/data.txt', 'w') as f:
    for i in range(len(file_paths)):
        folder1 = file_paths[i].split('\\')[-2]
        if i + 1 < len(file_paths):
            folder2 = file_paths[i + 1].split('\\')[-2]
            f.write(
                f'{file_paths[i]} {file_paths[i + 1]} {int(not folder1 == folder2)}\n')
        random_path = random.choice(file_paths)
        folder3 = random_path.split('\\')[-2]
        f.write(
            f'{file_paths[i]} {random_path} {int(not folder1 == folder3)}\n')
