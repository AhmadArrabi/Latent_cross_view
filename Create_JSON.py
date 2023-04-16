import json
import os

SOURCE_DIR = "/gpfs2/scratch/aarrabi/semantic maps/"
TARGET_DIR = "/gpfs2/scratch/xzhang31/CVUSA/dataset/streetview/panos/"
PROMPT = "High quality street view panorama image"
JSON_DIR = "/gpfs2/scratch/aarrabi/ControlNet/Data/"

imgs_source = os.listdir(SOURCE_DIR)
imgs_target = os.listdir(TARGET_DIR)

imgs_source.sort()
imgs_target.sort()

assert len(imgs_source)==len(imgs_target), "Source and target dataset lengths do not match"

temp = {}
all_jsons = []
for i in range(len(imgs_source)):
    temp['source'] = SOURCE_DIR+ imgs_source[i]
    temp['target'] = TARGET_DIR+ imgs_target[i]
    temp['prompt'] = PROMPT

    with open(f'{JSON_DIR}prompt.txt', 'a') as f:
        f.write(json.dumps(temp))
        f.write('\n')

os.rename(f'{JSON_DIR}prompt.txt',f'{JSON_DIR}prompt.json')

