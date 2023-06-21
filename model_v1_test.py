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
#resume_path = './models/control_sd15_ini.ckpt'
model = create_model('./models/cldm_v15_2.yaml').to('cuda')
#model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model.learning_rate = 0.01
model.sd_locked = True
model.only_mid_control = False

for p in model.control_model.named_parameters():
    print(p[0], p[1].requires_grad)

dataset = VIGOR(mode="train", same_area=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
#batch = next(iter(dataloader))

trainer = pl.Trainer(gpus=1, precision=16, strategy="ddp", min_epochs=1, max_epochs=2)
print(trainer.fit(model, dataloader))

#model.control_model.latent_size = (2, 320, 64, 64)

#dic = {
#    'c_crossattn': [torch.randn(2, 77, 768).to('cuda')], 
#    'c_concat'   : [torch.randn(2, 14, 3, 512, 512).to('cuda')],
#    'c_seq_len'  : [torch.randn(2).to('cuda')],
#    'c_seq_pos'  : [torch.randn(2, 14, 2).to('cuda')],
#    'c_seq_mask' : [torch.randn(2, 14).to('cuda')]
#} 

#model.control_model(dic)
#transforms.ToPILImage()(batch['hint'][0,0,]).save('first1_seq_sample_5.png')
#transforms.ToPILImage()(batch['hint'][0,1,]).save('first2_seq_sample_5.png')
#transforms.ToPILImage()(batch['hint'][0,2,]).save('first3_seq_sample_5.png')
#transforms.ToPILImage()(batch['hint'][0,3,]).save('first4_seq_sample_5.png')
#transforms.ToPILImage()(batch['hint'][0,4,]).save('first5_seq_sample_5.png')
#transforms.ToPILImage()(batch['jpg'][0]).save('aerial_sample_5.png')
#print(batch['len'][0])
#print(batch['delta'][0])
#d=batch['delta'][0]
#m = torch.ones(d[:,0].shape)*320.0
#print(d,m)
#print(m - d[:,1], m + d[:,0])

#batch['jpg'] = einops.rearrange(batch['jpg'], 'b c h w -> b h w c')
#batch['hint'] = batch['hint'][:,0,:,:,:]
#batch['hint'] = einops.rearrange(batch['hint'], 'b c h w -> b h w c')
 
#x = torch.randn(2, 4, 64, 64).to('cuda')

#model.configure_optimizers()
#model.disable_SD()

#draw_graph(model, input_data=(x, dic), save_graph=True, filename='forwardgraph FULL MODEL')

#logger = ImageLogger(batch_frequency=50, local_dir='PRINTING STUFF')
#trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], strategy="ddp", min_epochs=1, max_epochs=2)
#print(trainer.fit(model, dataloader))

# Calculate dummy gradients
#out = model(x, dic)[0].mean().backward()
#s = dict(list(model.named_parameters()))
#
#for key, item in s.items():
#    print(key)
#    if(item.requires_grad):
#        print('requiresed grad')
#    else: print('does not requiresed grad')
#    if(item.grad is None):
#        print('This is also None')
#    else: print('NOT None')


#make_dot(out, params=dict(list(model.named_parameters()))).render("backprob graph FULL MODEL", format="png")

#grads = {}
#for name, param in model.named_parameters():
#    print(name)
#    if(param.requires_grad):
#        print('!!!!!!')
#        grads[f'{name}'] = param.grad.view(-1)
#print(grads)

#print(model.training_step(batch, 1))
#x, dic = model.get_input(batch, 'jpg')
#for key, value in dic.items():
#    print(key, value[0].shape)
#print(x.shape, '*'*50) #[batch, 4, 64, 64]
#t = torch.randint(0, 1000, (x.shape[0],), device='cuda').long()
#eps = model.apply_model(x, t, dic)  

   
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
