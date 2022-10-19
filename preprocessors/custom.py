import os
import shutil
import math
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str)
parser.add_argument("--target_path", type=str)
opts = parser.parse_args()

target_path = opts.target_path

os.makedirs(target_path, exist_ok=True)

tag_dirs = os.listdir(opts.img_path)

for tag_dir in tag_dirs:
    attribute_dirs = os.listdir(os.path.join(opts.img_path, tag_dir))
    for attribute_dir in attribute_dirs:
        open(os.path.join(target_path, f'{tag_dir}_{attribute_dir}.txt'), 'w')
        images = os.listdir(os.path.join(opts.img_path, tag_dir, attribute_dir))
        for image in images:
            if os.path.isfile(os.path.join(opts.imgs, tag_dir, attribute_dir, image)):
                with open(os.path.join(target_path, f'{tag_dir}_{attribute_dir}.txt'), mode='a') as f:
                    f.write(f'{os.path.abspath(os.path.join(opts.imgs, tag_dir, attribute_dir, image))} 0\n')
