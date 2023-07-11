from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse
from VIGOR_dataset_layout import *
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--logger_freq", type=int, default=300, help="logger frequency")
    parser.add_argument("--gpu", type=int, default=1, help="number of gpus in training")
    parser.add_argument("--min_epoch", type=int, default=1, help="minimum epochs")
    parser.add_argument("--max_epoch", type=int, default=10, help="maximum epochs")
    parser.add_argument("--exp_name", type=str, default="default exp", help="experiment name")

    opt = parser.parse_args()

    # Configs
    resume_path = './models/control_sd15_ini.ckpt'
    #resume_path = './lightning_logs/version_10908718/checkpoints/epoch=7-step=1503.ckpt'
    batch_size = opt.batch_size
    logger_freq = opt.logger_freq
    learning_rate = opt.lr
    sd_locked = True
    only_mid_control = False
    gpu = opt.gpu
    min_epoch, max_epoch = opt.min_epoch, opt.max_epoch
    exp = opt.exp_name

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    print('batch: ', batch_size, '\nsd locked: ', sd_locked, '\nonly mid control: ', only_mid_control,'\nlr: ', learning_rate,'\nlogger freq: ', logger_freq, '\ngpus: ', gpu,'\nmin epoch: ', min_epoch,'\nmax epoch: ', max_epoch,'\nname: ', exp)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./chpts/{exp}/',
        monitor=None,  
        save_top_k=-1, 
        save_last=True,  
        every_n_train_steps=2000
        )
    
    # Misc
    dataset = VIGOR()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, local_dir=exp)
    trainer = pl.Trainer(gpus=gpu, precision=32, callbacks=[logger], strategy="ddp", min_epochs=min_epoch, max_epochs=max_epoch)

    # Train!
    trainer.fit(model, dataloader)

