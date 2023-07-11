import torch 
import torch.nn as nn
from VIGOR_dataset_geomap import *
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torchvision.utils import save_image
import torch
import kornia
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class Discriminator(nn.Module):
    def __init__(self,
                img_size=[512, 512],
                device='cuda',
                lr=3e-4):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.lr=lr

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=1, device=device),
            nn.Sigmoid()
        )

        self.configure_optimizer()
        self.configure_loss()
        
    def forward(self, x):
        return self.cnn(x)
    
    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def configure_loss(self):
        self.loss = nn.BCEWithLogitsLoss() 

    def configure_scheduler(self, warmup_steps, t_total):
        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps, t_total)
    
    def get_loss(self, generated, actual):
        return self.loss(generated, actual)
    
    def step(self, generated, actual):
        self.optimizer.zero_grad()
        batch_size = generated.shape[0]
        
        ones = torch.full((batch_size,),  1., dtype=torch.float, device=self.device)
        zeros = torch.full((batch_size,),  0., dtype=torch.float, device=self.device)

        real_pred = torch.sum(self(actual), (1,2,3))
        real_loss = self.get_loss(real_pred, ones)
        real_loss.backward(retain_graph=True)

        fake_pred = torch.sum(self(generated.detach()), (1,2,3))
        fake_loss = self.get_loss(fake_pred, zeros)
        fake_loss.backward(retain_graph=True)
        
        self.optimizer.step()

class GeoNet(nn.Module):
    def __init__(self,
                seq_padding=14,
                img_size=[512, 512],
                device='cuda',
                lr=3e-4,
                VAE_config_path='./VAE_geomap.yaml',
                VAE_ckpt_path='./models/VAE.ckpt',
                create_ckpt=False):
        super().__init__()
        self.seq_padding = seq_padding
        self.img_size = img_size
        self.device = device
        self.create_ckpt=create_ckpt
        self.lr=lr

        self.backbone = self.instantiate_VAE(VAE_config_path, VAE_ckpt_path, create_ckpt)
        self.discriminator = Discriminator(img_size=img_size, device=device, lr=lr)

        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1, device=self.device),
            nn.ReLU()
        )

        #self.convs = nn.Sequential(
        #    nn.Conv2d(self.seq_padding*4, 64, kernel_size=3, stride=2, padding=1, device=self.device),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, device=self.device),
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, device=self.device),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, device=self.device),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1, device=self.device),
        #    nn.ReLU()
        #)
    
    def instantiate_VAE(self, config_path, ckpt_path, create_ckpt=False, SD_ckpt='./models/v1-5-pruned.ckpt'):
        model = self.create_model(config_path=config_path).to(self.device)

        if create_ckpt:
            scratch_dict = model.state_dict()
            target_dict = {}

            pretrained_weights = torch.load(SD_ckpt)
            if 'state_dict' in pretrained_weights:
                pretrained_weights = pretrained_weights['state_dict']

            for k in scratch_dict.keys():
                if f'first_stage_model.{k}' in pretrained_weights:
                    print(f'{k} IS TRANSFERRED')
                    target_dict[k] = pretrained_weights[f'first_stage_model.{k}'].clone()
                else:
                    print(f'WARNING: {k} WAS NOT FOUND')

            model.load_state_dict(target_dict, strict=True)
            torch.save(model.state_dict(), './models/VAE.ckpt')
            print('Done.')

        else:
            pretrained_weights = torch.load(ckpt_path)
            model.load_state_dict(pretrained_weights, strict=True)
        
        return model

    def create_model(self, config_path):
        config = OmegaConf.load(config_path)
        model = self.disable_VAE(config.model)
        print(f'Loaded model config from [{config_path}]')
        return model
        
    def disable_VAE(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False #maybe play with this
        return model
    
    def geo_mapping(self, hint_latent, seq_pos, B):
        desired_height = hint_latent.shape[2]*2
        desired_width = hint_latent.shape[3]*2

        vertical_pad = (desired_height - hint_latent.shape[2])//2
        horizontal_pad = (desired_width - hint_latent.shape[3])//2

        hint_padded = torch.nn.functional.pad(hint_latent, (horizontal_pad, horizontal_pad, vertical_pad, vertical_pad)).to(self.device)
        
        latent_ratio_x = desired_width/self.img_size[1]
        latent_ratio_y = desired_height/self.img_size[0]

        x_shift = (seq_pos[:,0]*latent_ratio_x).round()
        y_shift = (seq_pos[:,1]*latent_ratio_y).round()

        affine_matrix = torch.tensor([[1, 0, 0.0], [0, 1, 0]], dtype=hint_padded.dtype).to(self.device)
        affine_matrix = affine_matrix.unsqueeze(0).repeat(B*self.seq_padding, 1, 1)   #for each seq

        affine_matrix[:,0,-1] = x_shift
        affine_matrix[:,1,-1] = -y_shift

        return kornia.geometry.transform.warp_affine(hint_padded, affine_matrix, dsize=(desired_height, desired_width))
    
    def forward(self, ground, position):#mask
        batch_size = ground.shape[0] #ground = [2, 14, 3, 512, 1024]
        if batch_size == self.seq_padding: batch_size=1
        seq_ground = ground.view(batch_size*self.seq_padding, 3, self.img_size[0], self.img_size[1]) #[2*14, 3, 512, 1024]
        #seq_pos = position.view(batch_size*self.seq_padding, 2)

        z = self.backbone.encode(seq_ground).sample() #[2*14, 4, 64, 64]
        #z = self.geo_mapping(z, seq_pos, batch_size) #[2*14, 4, 128, 128] 
        #z = z.view(batch_size, self.seq_padding, 4, self.img_size[0]//4, self.img_size[1]//4) #[2, 14, 4, 128, 128

        #z = z*mask[(..., ) + (None, ) * 3] 
        #z = z.view(batch_size, self.seq_padding*4, self.img_size[0]//4, self.img_size[1]//4) #[2, 14*4, 128, 128]

        z = self.convs(z)

        return z #[2, 4, 64, 64]
    
    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.convs.parameters(), lr=self.lr)

    def configure_loss(self):
        self.loss = nn.SmoothL1Loss() #nn.KLDivLoss(reduction="batchmean")
        self.adversarial_loss_f = nn.BCEWithLogitsLoss()

    def configure_scheduler(self, warmup_steps, t_total):
        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps, t_total)
    
    def get_loss(self, z_pred, z_layout, generated):
        #return self.loss(z_pred, z_layout)
        L1loss = self.loss(z_pred, z_layout)

        ones = torch.full((generated.shape[0],),  1., dtype=torch.float, device=self.device)

        t = torch.sum(self.discriminator(generated), (1,2,3))
        adversarial_loss = self.adversarial_loss_f(t, ones)
        print('L1 loss: ', L1loss, 'Adverserial loss: ', adversarial_loss)

        return (0.8*L1loss + 0.2*adversarial_loss)/2
    
    def step(self, batch, i, e):
        self.optimizer.zero_grad()
        
        layout = batch['jpg'].to(self.device)
        ground = batch['hint'].to(self.device)
        position = batch['delta'].to(self.device)

        z_pred = self(ground, position)#, mask)
        z_layout = self.backbone.encode(layout).sample()

        self.log(i, e, z_pred, z_layout)
        #self.log_gnd(i, ground, gnd_len)

        generated = self.backbone.decode(z_pred)
        self.discriminator.step(generated=generated, actual=layout)

        loss = self.get_loss(z_pred, z_layout, generated)
        loss.backward()

        self.optimizer.step()
    
    def log(self, i, e, z_pred, z_layout):
        if i%self.freq==0:
            save_image(self.backbone.decode(z_pred)[0], f'./{self.log_file}/reconstruction_e{e}_s{i}.png')
            save_image(self.backbone.decode(z_layout)[0], f'./{self.log_file}/real layout_e{e}_s{i}.png')
        
    def log_gnd(self, i, ground, gnd_len):
        if i%self.freq==0:
            for m, seq in enumerate(gnd_len):
                for s in range(seq):
                    save_image(ground[m,s,:], f'./{self.log_file}/reconstruction_{m}_{s}.png')

    def train(self, epoch, training_loader, log_freq, warmup_steps, log_file):
        self.freq = log_freq
        self.log_file=log_file
        self.configure_optimizer()
        self.configure_loss()
        self.configure_scheduler(warmup_steps, epoch)
        self.discriminator.configure_scheduler(warmup_steps, epoch)

        for e in tqdm(range(epoch)):
            for i, batch in enumerate(training_loader):
                self.step(batch, i, e)
            self.scheduler.step()
            self.discriminator.scheduler.step()

dataset = VIGOR()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
mod = GeoNet(img_size=[256,256], seq_padding=1)
mod.train(epoch=20, training_loader=dataloader, log_freq=50, warmup_steps=3, log_file='log_no_geo')






