import json
import os

mode = 'train'

CSV_splits = f"/gpfs2/scratch/xzhang31/CVUSA/dataset/splits/{mode}-19zl.csv"
SOURCE_DIR = "/gpfs2/scratch/xzhang31/CVUSA/dataset/"
TARGET_DIR = "/gpfs2/scratch/xzhang31/CVUSA/dataset/"
PROMPT = "Photo-realistic aerial-view image with high quality details."
JSON_DIR = "./Data/"

with open(f'{JSON_DIR}{mode}_prompt.txt', 'a') as f:
    with open(CSV_splits, 'r') as p:
        for i in p.readlines():
            temp_dict = {}
            i = i.strip().split(',')
            temp_dict['source'] = os.path.join(SOURCE_DIR, i[1])
            temp_dict['target'] = os.path.join(TARGET_DIR, i[1])
            temp_dict['prompt'] = PROMPT
            f.write(json.dumps(temp_dict))
            f.write('\n')


# with open(f'{JSON_DIR}train_prompt.txt', 'a') as f:
# for i in range(len(imgs_source)):
#     temp['source'] = SOURCE_DIR + imgs_source[i]
#     temp['target'] = TARGET_DIR + imgs_target[i]
#     temp['prompt'] = PROMPT

#     f.write(json.dumps(temp))
#     f.write('\n')

# os.rename(f'{JSON_DIR}prompt.txt',f'{JSON_DIR}prompt.json')

