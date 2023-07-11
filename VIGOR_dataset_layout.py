import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
from torch.utils.data import DataLoader
import torchvision
import json
import einops
from ldm.util import *

def aerial_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],
                   std=[.5,.5,.5])
    ])
    
def layout_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],
                    std=[.5,.5,.5])
    ])

class VIGOR(torch.utils.data.Dataset):
    def __init__(self, root = '/gpfs2/scratch/aarrabi/ControlNet/Seattle/satellite/', args=None):
        super(VIGOR, self).__init__()

        self.args = args
        self.root = root
        
        # The below size is temporary should check later
        self.sat_size = [512, 512]
        self.layout_size = [512, 512]

        # transforms notice strong aug is added
        self.transform_aerial = aerial_transform(size=self.sat_size)
        self.transform_layout = layout_transform(size=self.layout_size)

        self.image_names = os.listdir(self.root)

        
    def __getitem__(self, index):
        prompt = 'Realistic aerial satellite top view image with high quality details, with buildings, trees, and roads in snowy wethear with snow'
        image_name = self.image_names[index]
        
        layout_path = os.path.join(self.root, image_name)
        aerial_path = os.path.join('/gpfs2/scratch/xzhang31/VIGOR/Seattle/satellite/', image_name)

        layout_image = Image.open(layout_path, 'r').convert('RGB')
        aerial_image = Image.open(aerial_path, 'r').convert('RGB')
    
        layout_image = self.transform_aerial(layout_image)
        aerial_image = self.transform_aerial(aerial_image)
        layout_image = einops.rearrange(layout_image, 'c h w -> h w c')
        aerial_image = einops.rearrange(aerial_image, 'c h w -> h w c')
        
        return dict(jpg=aerial_image, 
                    txt=prompt, 
                    hint=layout_image)

    def __len__(self):
        return len(self.image_names)

if __name__ == "__main__":

    dataset = VIGOR()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    
    idx = 0
    for i in dataloader:
        idx += 1
        
        print(idx, "#"*30)
        print('aerial shape: ', i['jpg'].shape, '\nhint shape: ', i['hint'].shape, '\ntext: ', i['txt'])
        
        if idx==5: break
            
