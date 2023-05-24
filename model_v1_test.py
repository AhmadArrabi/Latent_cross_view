from share import *
#import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from CVUSA_dataset import MyDataset
#from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
#import argparse
from VIGOR_dataset import *
import einops

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
resume_path = './models/control_sd15_ini.ckpt'
model = create_model('./models/cldm_v15.yaml').to('cuda')
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
dataset = VIGOR(mode="train", same_area=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
batch = next(iter(dataloader))
batch['jpg'] = einops.rearrange(batch['jpg'], 'b c h w -> b h w c')
#batch['hint'] = batch['hint'][:,0,:,:,:]
#batch['hint'] = einops.rearrange(batch['hint'], 'b c h w -> b h w c')
for key, value in batch.items():
    if key != 'txt':
        value.to('cuda')
x, dic = model.get_input(batch, 'jpg')
#t = torch.randint(0, 1000, (x.shape[0],), device='cuda').long()

#eps = model.apply_model(x, t, dic)       

#print(x, dic)

"""
aerial shape:  torch.Size([4, 3, 640, 640])
hint shape  :  torch.Size([4, 14, 3, 1024, 2048])
delta hsape :  torch.Size([4, 14, 2])
number of ground:  tensor([2, 1, 2, 3])
text:  ['Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.']
"""

