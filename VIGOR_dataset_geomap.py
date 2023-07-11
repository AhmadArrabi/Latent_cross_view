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

def layout_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],
                    std=[.5,.5,.5])
    ])

def ground_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],
                    std=[.5,.5,.5])
    ])

class VIGOR(torch.utils.data.Dataset):
    def __init__(self, root = '/gpfs3/scratch/xzhang31/VIGOR', same_area=True, args=None):
        #'/gpfs3/scratch/aarrabi/VIGOR'
        super(VIGOR, self).__init__()

        self.args = args
        self.root = root

        self.seq_padding = 6
        
        # The below size is temporary should check later
        self.layout_size = [256, 256]
        self.grd_size = [256, 256] 

        # transforms notice strong aug is added
        self.transform_ground = ground_transform(size=self.grd_size)
        self.transform_layout = layout_transform(size=self.layout_size)

        label_root = 'splits__corrected'
        self.layout_root = '/gpfs2/scratch/aarrabi/ControlNet/Seattle/satellite/'
        #self.layout_root2 = '/gpfs2/scratch/aarrabi/ControlNet/'

        self.train_city_list = ['Seattle']

        self.train_dict = {} # mapping between aerial images to ground images
        self.train_list = [] # aerial images list

        for c in self.train_city_list:
            json_file = 'same_area_balanced_train__corrected.json'
            with open(os.path.join(self.root, label_root, f'{c}_AerialMajorSplit', json_file), 'r') as j:
                city_dict = json.load(j)
                for k in city_dict.keys():
                    self.train_list.append(k)
                self.train_dict = {**self.train_dict, **city_dict}
        
        self.layout_list = os.listdir(self.layout_root)
        #self.layout_list = [f'Seattle/satellite/{i}' for i in self.layout_list]
        self.train_dict = {key: self.train_dict[key] for key in self.layout_list}

    def __getitem__(self, index):
        layout_image_name = self.layout_list[index]
        ground_dict = self.train_dict[layout_image_name]

        temp_img = Image.open(os.path.join(self.layout_root,layout_image_name), 'r').convert('RGB')
        
        DELTA_SCALE = self.grd_size[0]/temp_img.size[0] #new/old dimensions, assuming square images (which is a true assumption :)
        
        layout_image = self.transform_layout(temp_img)
        
        ground_image_list = []
        ground_delta_list = []
        num_g_imgs = len(ground_dict)

        for k,v in ground_dict.items():
            ground_image_list.append(
                self.transform_ground(
                    Image.open(os.path.join(self.root, 'Seattle/panorama/',k).replace('png', 'jpg'), 'r').convert('RGB')
                )
            )
            ground_delta_list.append([-float(v[1])*DELTA_SCALE, -float(v[0])*DELTA_SCALE])

        # padding
        if num_g_imgs<=5:
            zero_tensor = torch.zeros(ground_image_list[0].shape)
            remaining_dummy_tensors =self.seq_padding - num_g_imgs
            for i in range(remaining_dummy_tensors):
                ground_image_list.append(zero_tensor)
                ground_delta_list.append([0.0, 0.0])
            
            ground_imgs = torch.cat(ground_image_list, dim=0)
            ground_imgs = ground_imgs.reshape(shape=(self.seq_padding, 3, ground_imgs.shape[1], ground_imgs.shape[2]))
        else:
            ground_imgs = torch.cat(ground_image_list, dim=0)
            ground_imgs = ground_imgs.reshape(shape=(num_g_imgs, 3, ground_imgs.shape[1], ground_imgs.shape[2]))
        
        ground_deltas = torch.tensor(ground_delta_list)

        mask = torch.zeros(self.seq_padding)
        mask[:num_g_imgs]=1

        ground_imgs = ground_imgs[:self.seq_padding]
        ground_deltas = ground_deltas[:self.seq_padding]
        mask = mask[:self.seq_padding]
        
        return dict(jpg=layout_image, 
                    hint=ground_imgs, 
                    delta=ground_deltas)#, 
                    #len=num_g_imgs,
                    #mask=mask)
    
    def __len__(self):
        return len(self.train_dict)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

if __name__ == "__main__":

    dataset = VIGOR()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    
    idx = 0
    for i in dataloader:
        idx += 1
        if idx == 10:
            #tensor_to_image(i['jpg'][0]).save("aerial.png")
            #grounds = torch.split(i['hint'][0], 1)
            
            #for num in range(i['len'][0]):
            #    ass = einops.rearrange(grounds[num].squeeze(), 'c h w -> h w c')
            #    tensor_to_image(ass).save(f"ground_{num}.png")

            #num_channel = i['hint'].shape[1]
            #num_images = int(num_channel) // 3
            #grd_images = i['hint'].reshape(num_images, 3, i['hint'].shape[2], i['hint'].shape[3])
            #torchvision.utils.save_image(i['hint'][0], f"grd{idx}.png")
            print(idx, "#"*30)
            print('aerial shape: ', i['jpg'].shape,'\nhint (ground tensor) shape: ', i['hint'].shape, '\ndelta ', i['delta'].shape, '\nnumber of ground: ')
            break
            
