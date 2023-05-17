from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from CVUSA_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse

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

    opt = parser.parse_args()

    # Configs
    resume_path = './models/control_sd15_ini.ckpt'
    batch_size = opt.batch_size
    logger_freq = opt.logger_freq
    learning_rate = opt.lr
    sd_locked = opt.sd_locked
    only_mid_control = opt.only_mid_control
    gpu = opt.gpu
    min_epoch, max_epoch = opt.min_epoch, opt.max_epoch

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset(mode="train")
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=gpu, precision=32, callbacks=[logger], strategy="ddp", min_epochs=min_epoch, max_epochs=max_epoch)

    # Train!
    trainer.fit(model, dataloader)
