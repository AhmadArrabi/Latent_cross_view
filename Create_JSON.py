import json
import os

mode = 'val'
CSV_splits = f"/gpfs2/scratch/xzhang31/CVUSA/dataset/splits/{mode}-19zl.csv"
task = 'sem2street' #if we want g2a change to 'street2aerial'

if task == 'sem2street':
    SOURCE_DIR = "/gpfs2/scratch/aarrabi/semantice maps/"
    TARGET_DIR = "CVUSA/dataset/"
    PROMPT = "Photo-realistic street-view panorama image with high quality details"
    JSON_DIR = "./Data/sem2street/"

    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)
        print(f"File path {JSON_DIR} created as it did not exist :(")

    with open(f'{JSON_DIR}{mode}_prompt.txt', 'w') as f:
        with open(CSV_splits, 'r') as p:
            for i in p.readlines():
                temp_dict = {}
                i = i.strip().split(',')
                temp_dict['source'] = os.path.join(SOURCE_DIR, f"seg{i[1].split('/')[-1]}")
                temp_dict['target'] = os.path.join(TARGET_DIR, i[1])
                temp_dict['prompt'] = PROMPT
                f.write(json.dumps(temp_dict))
                f.write('\n')

elif task == 'street2aerial':
    SOURCE_DIR = "CVUSA/dataset/"
    TARGET_DIR = "CVUSA/dataset/"
    PROMPT = "Photo-realistic aerial-view image with high quality details."
    JSON_DIR = "./Data/street2aerial/"

    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)
        print(f"File path {JSON_DIR} created as it did not exist :(")

    with open(f'{JSON_DIR}{mode}_prompt.txt', 'w') as f:
        with open(CSV_splits, 'r') as p:
            for i in p.readlines():
                temp_dict = {}
                i = i.strip().split(',')
                temp_dict['source'] = os.path.join(SOURCE_DIR, i[1])
                temp_dict['target'] = os.path.join(TARGET_DIR, i[0])
                temp_dict['prompt'] = PROMPT
                f.write(json.dumps(temp_dict))
                f.write('\n')

else: raise Exception(f'task {task} not yet implemented please choose sem2street or street2aerial')