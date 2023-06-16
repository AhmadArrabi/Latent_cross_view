from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse
from VIGOR_dataset import *
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--logger_freq", type=int, default=300, help="logger frequency")
    parser.add_argument("--sd_locked", type=bool, default=True, help="lock stable diffusion")
    parser.add_argument("--only_mid_control", type=bool, default=False, help="only control encoder")
    parser.add_argument("--gpu", type=int, default=1, help="number of gpus in training")
    parser.add_argument("--min_epoch", type=int, default=1, help="minimum epochs")
    parser.add_argument("--max_epoch", type=int, default=10, help="maximum epochs")
    parser.add_argument("--exp_name", type=str, default="default exp", help="experiment name")

    opt = parser.parse_args()

    # Configs
    resume_path = './models/control_sd15_ini_zeroConv_unFreeze.ckpt'
    batch_size = opt.batch_size
    logger_freq = opt.logger_freq
    learning_rate = opt.lr
    sd_locked = opt.sd_locked
    only_mid_control = opt.only_mid_control
    gpu = opt.gpu
    min_epoch, max_epoch = opt.min_epoch, opt.max_epoch
    exp = opt.exp_name

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15_2.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./chpts/{exp}/',
        monitor=None,  
        save_top_k=-1, 
        save_last=True,  
        every_n_train_steps=2000  
        )
    
    # Misc
    dataset = VIGOR(mode="train", same_area=True)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    logger = ImageLogger(batch_frequency=logger_freq, local_dir=exp)
    trainer = pl.Trainer(gpus=gpu, precision=16, callbacks=[logger, checkpoint_callback], strategy="ddp", min_epochs=min_epoch, max_epochs=max_epoch)

    # Train!
    trainer.fit(model, dataloader)

