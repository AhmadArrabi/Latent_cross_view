from share import *
#import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from CVUSA_dataset import MyDataset
#from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlSeq
#import argparse
from VIGOR_dataset import *
import einops
import torch.nn as nn
import math
from torchviz import make_dot
from torchview import draw_graph
import hiddenlayer as hl
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
resume_path = './models/control_sd15_ini.ckpt'
model = create_model('./models/cldm_v15.yaml').to('cuda')
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
dataset = VIGOR(mode="train", same_area=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
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
latent_size = x.shape
transformation, latent_ratio = 1,1
seq_padding = 14
model_channels = 320
mod = ControlSeq(latent_size = latent_size, transformation = transformation, seq_padding = seq_padding, latent_ratio = latent_ratio, model_channels = model_channels).to('cuda')
out = mod.forward(dic)
loss_fn = nn.MSELoss()
test_out = torch.cat(out)
test_target = torch.randn(test_out.shape)
loss = loss_fn(test_target.to('cuda'), test_out.to('cuda'))
make_dot(out[2], params=dict(list(mod.named_parameters()))).render("backprob graph", format="png")
draw_graph(mod, input_data=[dic], save_graph=True, filename='forwardgraph')
#im = hl.build_graph(mod, dic)
#im.save(path="../forwardgraph" , format="png")
loss.backward()
print(dic['c_concat'][0].grad, '*'*80, '\ln')
print(dic['c_seq_len'][0].grad, '*'*80, '\ln')
print(dic['c_seq_pos'][0].grad, '*'*80, '\ln')
print(test_out.grad, '*'*80, '\ln')
#print(x, dic)
"""
aerial shape:  torch.Size([4, 3, 640, 640])
hint shape  :  torch.Size([4, 14, 3, 1024, 2048])
delta hsape :  torch.Size([4, 14, 2])
number of ground:  tensor([2, 1, 2, 3])
text:  ['Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.']
"""

