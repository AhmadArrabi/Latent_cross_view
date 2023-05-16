import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
from torch.utils.data import DataLoader
#from .augmentations import HFlip, Rotate
import torchvision
import json

class LimitedFoV(object):

    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x

        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size, mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomPosterize(p=0.2, bits=4),
            transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    elif "test" in mode:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise RuntimeError(f"{mode} not implemented")

# Same loader from VIGOR, modified for pytorch
class VIGOR(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', root = '/gpfs2/scratch/xzhang31/VIGOR', same_area=True, args=None):
        super(VIGOR, self).__init__()

        self.args = args
        self.root = root

        self.mode = mode
        if self.mode not in ['train', 'test']:
            raise RuntimeError(f'{self.mode} is not implemented!')
        
        # The below size is temporary should check later
        self.sat_size = [320, 320]
        self.sat_size_default = [320, 320]
        self.grd_size = [128, 512]

        # transforms notice strong aug is added
        self.transform_ground = input_transform(size=self.grd_size, mode=self.mode)
        self.transform_aerial = input_transform(size=self.sat_size, mode=self.mode)

        self.same_area = same_area
        label_root = 'splits__corrected'

        if same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        self.train_dict = {} # mapping between aerial images to ground images
        self.train_list = [] # aerial images list
        for c in self.train_city_list:
            json_file = 'same_area_balanced_train__corrected.json' if same_area else "pano_label_balanced__corrected.json"
            with open(os.path.join(self.root, label_root, f'{c}_AerialMajorSplit', json_file), 'r') as j:
                city_dict = json.load(j)
                for k in city_dict.keys():
                    self.train_list.append(k)
                self.train_dict = self.train_dict | city_dict # aggregate dictionaries
        
        self.test_dict = {}
        self.test_list = []
        for c in self.test_city_list:
            json_file = 'same_area_balanced_test__corrected.json' if same_area else "pano_label_balanced__corrected.json"
            with open(os.path.join(self.root, label_root, f'{c}_AerialMajorSplit', json_file), 'r') as j:
                city_dict = json.load(j)
                for k in city_dict.keys():
                    self.test_list.append(k)
                self.test_dict = self.test_dict | city_dict

    def __getitem__(self, index, debug=False):
        #TODO
        # Implement random sampled center in aerial images
        # Implement LS aug (rotate, flip)
        if self.mode == 'train':
            prompt = 'Photo-realistic aerial-view image with high quality details.'
            aerial_image_name = self.train_list[index]
            ground_dict = self.test_dict[aerial_image_name]

            print(aerial_image_name)
            
            aerial_image = self.transform_aerial(Image.open(aerial_image_name))
            
            ground_image_list = []
            ground_delta_list = []
            num_g_imgs = len(ground_dict)
            for k,v in ground_dict.items():
                ground_image_list.append(
                    self.transform_ground(
                        Image.open(k)
                    )
                )
                ground_delta_list.append([int(v[0]), int(v[1])])
                
            ground_imgs = torch.cat(ground_image_list, dim=0)
            ground_deltas = torch.tensor(ground_delta_list)
            
            return dict(jpg=aerial_image, 
                        txt=prompt, 
                        hint=ground_imgs, 
                        delta=ground_deltas, 
                        len=num_g_imgs)
            
        
        elif self.mode == 'test':
            prompt = 'Photo-realistic aerial-view image with high quality details.'
            aerial_image_name = self.test_list[index]
            ground_dict = self.test_dict[aerial_image_name]

            print(aerial_image_name)
            
            aerial_image = self.transform_aerial(Image.open(aerial_image_name))

            ground_image_list = []
            ground_delta_list = []
            num_g_imgs = len(ground_dict)
            for k,v in ground_dict.items():
                ground_image_list.append(
                    self.transform_ground(
                        Image.open(k)
                    )
                )
                ground_delta_list.append([int(v[0]), int(v[1])])
                
            ground_imgs = torch.cat(ground_image_list, dim=0)
            ground_deltas = torch.tensor(ground_delta_list)
            
            return dict(jpg=aerial_image, 
                        txt=prompt, 
                        hint=ground_imgs, 
                        delta=ground_deltas, 
                        len=num_g_imgs)
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        elif self.mode == 'test':
            return len(self.test_list)
        else:
            print('not implemented!')
            raise Exception


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C = np.sin(lat_A)*np.sin(lat_B) + np.cos(lat_A)*np.cos(lat_B)*np.cos(lng_A-lng_B)
    distance = R*np.arccos(C)
    return distance


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance_matrix(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C1 = np.matmul(np.sin(np.expand_dims(lat_A,axis=1)), np.sin(np.expand_dims(lat_B,axis=0)))
    C2 = np.matmul(np.cos(np.expand_dims(lat_A,axis=1)),np.cos(np.expand_dims(lat_B,axis=0)))
    C2 = C2 * np.cos(np.tile(np.expand_dims(lng_A,axis=1),[1,lng_B.shape[0]])-np.tile(lng_B,[np.expand_dims(lng_A,axis=0).shape[0],1]))
    C = C1 + C2
    distance = R*np.arccos(C)
    return distance


# compute the delta unit for each reference location [Lat, Lng], 320 is half of the image width
# 0.114 is resolution in meter
# reverse equation from gps2distance: https://en.wikipedia.org/wiki/Great-circle_distance
def Lat_Lng(Lat_A, Lng_A, distance=[320*0.114, 320*0.114]):
    if distance[0] == 0 and distance[1] == 0:
        return np.zeros(2)

    lat_A = Lat_A * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    R = 6371004.
    C_lat = np.cos(distance[0]/R)
    C_lng = np.cos(distance[1]/R)
    delta_lat = np.arccos(C_lat)
    delta_lng = np.arccos((C_lng-np.sin(lat_A)*np.sin(lat_A))/np.cos(lat_A)/np.cos(lat_A))
    return np.array([delta_lat * 180. / np.pi, delta_lng * 180. / np.pi])


if __name__ == "__main__":
    dataset = VIGOR(mode="train", same_area=True)
    # dataset = VIGOR(mode="test_query",root = '/mnt/VIGOR/', same_area=True, print_bool=True)
    # dataset = VIGOR(mode="test_reference",root = '/mnt/VIGOR/', same_area=True, print_bool=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    idx = 0
    for i in dataloader:
        torchvision.utils.save_image(i[0], "grd_f.png")
        torchvision.utils.save_image(i[1], "grd_s.png")
        torchvision.utils.save_image(i[2], "sat_f.png")
        torchvision.utils.save_image(i[3], "sat_s.png")
        if idx > 3:
            break
        idx += 1