import os
import json

import cv2
import numpy as np

from torch.utils.data import Dataset

# JSON_DIR = "./Data/prompt.txt"
TARGET_SIZE = 256
RESIZE_SCALE = 2.0   

class MyDataset(Dataset):
    def __init__(self, mode="train", data_dir='../scratch'):
        self.mode = mode
        self.data_dir = data_dir

        if self.mode == "train":
            prompt_dir = "./Data/train_prompt.txt"
        elif self.mode == "val":
            prompt_dir = "./Data/val_prompt.txt"
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

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

