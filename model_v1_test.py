from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from CVUSA_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
#from cldm.cldm import ControlSeq
#import argparse
from VIGOR_dataset import *
import einops
import torch.nn as nn
import torch
#import math
from torchviz import make_dot
from torchview import draw_graph

#First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
resume_path = './models/control_sd15_ini_2.ckpt'
model = create_model('./models/cldm_v15_2.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = 0.00001
model.sd_locked = True
model.only_mid_control = False

dataset = VIGOR(mode="train", same_area=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
batch = next(iter(dataloader))
batch['jpg'] = einops.rearrange(batch['jpg'], 'b c h w -> b h w c')

#batch['hint'] = batch['hint'][:,0,:,:,:]
#batch['hint'] = einops.rearrange(batch['hint'], 'b c h w -> b h w c')

logger = ImageLogger(batch_frequency=4, local_dir='first run')
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], strategy="ddp", min_epochs=1, max_epochs=2)
print(trainer.fit(model, dataloader))

#print(model.training_step(batch, 1))
#x, dic = model.get_input(batch, 'jpg')
#for key, value in dic.items():
#    print(key, value[0].shape)
#print(x.shape, '*'*50) #[batch, 4, 64, 64]
#t = torch.randint(0, 1000, (x.shape[0],), device='cuda').long()
#eps = model.apply_model(x, t, dic)  

#dic = {
#    'c_crossattn': [torch.randn(2, 77, 768).to('cuda')], 
#    'c_concat'   : [torch.randn(2, 14, 3, 128, 256).to('cuda')],
#    'c_seq_len'  : [torch.randn(2).to('cuda')],
#    'c_seq_pos'  : [torch.randn(2, 14, 2).to('cuda')],
#    'c_seq_mask' : [torch.randn(2, 14).to('cuda')]
#}     
#
#transformation = 1
#seq_padding = 14
#model_channels = 320
#
#mod = ControlSeq(seq_padding=seq_padding, model_channels=model_channels, transformation=transformation, channel_mult=[1,2,4,4]).to('cuda')
#mod.latent_size = (2, 320, 64, 64)
#
#out = mod.forward(dic)
#loss_fn = nn.MSELoss()
#test_out = out[0]
#test_target = torch.randn(test_out.shape)
#loss = loss_fn(test_target.to('cuda'), test_out.to('cuda'))
#
#make_dot(out[0], params=dict(list(mod.named_parameters()))).render("backprob graph controlseq", format="png")
#draw_graph(mod, input_data=[dic], save_graph=True, filename='forwardgraph controlseq')
#
#loss.backward()
#print(mod.zero_convs[0].weight.grad, '*'*80, '\ln')
#print(mod.zero_convs[0].weight, '*'*80, '\ln')
#print(mod.zero_convs[5].weight.grad, '*'*80, '\ln')
#print(mod.zero_convs[5].weight, '*'*80, '\ln')
#print(dic['c_seq_len'][0].grad, '*'*80, '\ln')
#print(dic['c_seq_pos'][0].grad, '*'*80, '\ln')
#print(test_out.grad, '*'*80, '\ln')
#print(x, dic)

"""
aerial shape:  torch.Size([4, 3, 640, 640])
hint shape  :  torch.Size([4, 14, 3, 1024, 2048])
delta hsape :  torch.Size([4, 14, 2])
number of ground:  tensor([2, 1, 2, 3])
text:  ['Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.', 'Photo-realistic aerial-view image with high quality details.']
"""
