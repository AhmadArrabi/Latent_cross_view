import os
import json

import cv2
import numpy as np

from torch.utils.data import Dataset

# JSON_DIR = "./Data/prompt.txt"
TARGET_SIZE = 256
RESIZE_SCALE = 2.0   

class MyDataset(Dataset):
    def __init__(self, mode="train", data_dir='../scratch', task="semantic2street", blur_aug='normal'):
        self.mode = mode.lower()
        self.data_dir = data_dir
        self.task = task
        self.blur_aug= blur_aug.lower()

        if self.task == "semantic2street":
            parent_prompt_dir = "./Data/semantic2street/"
        elif self.task == "street2aerial":
            parent_prompt_dir = "./Data/street2aerial/"
        else: raise RuntimeError(f'task:{self.task} not yet implemented! please choose semantic2street or street2aerial')

        if self.mode == "train":
            prompt_dir = f"{parent_prompt_dir}train_prompt.txt"
        elif self.mode == "val":
            prompt_dir = f"{parent_prompt_dir}val_prompt.txt"
        else:
            raise RuntimeError(f"mode:{self.mode} is not implemented!")
        
        self.data = []
        with open(prompt_dir, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.data_dir, source_filename))
        target = cv2.imread(os.path.join(self.data_dir, target_filename))
        print(os.path.join(self.data_dir, source_filename))
        print(os.path.join(self.data_dir, target_filename))

        #reshape
        w_source, h_source = source.shape[1], source.shape[0]
        w_target, h_target = target.shape[1], target.shape[0]

        w_source = int(w_source / RESIZE_SCALE)
        w_target = int(w_target / RESIZE_SCALE)
        h_source = int(h_source / RESIZE_SCALE)
        h_target = int(h_target / RESIZE_SCALE)

        (w_source, h_source) = map(lambda x: x - x % 64, (w_source, h_source))  # resize to integer multiple of 64
        (w_target, h_target) = map(lambda x: x - x % 64, (w_target, h_target))  # resize to integer multiple of 64

        # w_source = 256
        # w_target = 256
        # h_source = 256
        # h_target = 256
        
        source = cv2.resize(source, (w_source, h_source), interpolation = cv2.INTER_AREA)
        target = cv2.resize(target, (TARGET_SIZE, TARGET_SIZE), interpolation = cv2.INTER_AREA)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        #random guassian blur and noise augmentation
        #initialize with no effect
        blur, std = 0,0
        if self.task == "semantic2street" and self.blur_aug == 'normal':
            blur = np.random.choice([0,7,21])
            std = np.random.uniform(low=0, high=0.5)
        elif self.task == "semantic2street" and self.blur_aug == 'strong':
            blur = np.random.choice([7,14,21])
            std = np.random.uniform(low=0, high=0.8)
        elif self.task == "semantic2street" and self.blur_aug == 'none':
            blur, std = 0, 0
        elif self.task == "semantic2street":
            raise RuntimeError(f"Augmentation {self.blur_aug} is not implemented! choose normal, strong, or none")

        if blur:
            source = cv2.GaussianBlur(source, (blur,blur), cv2.BORDER_DEFAULT)

        gauss = np.random.normal(0, std, source.shape).astype('uint8')
        source = cv2.add(source, gauss)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) #/ 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

