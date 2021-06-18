from argparse import ArgumentParser
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from h_CAE3d import CAE
from HIT_dataset3d import hit_dm

def main(hparams):
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=True,
    save_last=True,
    verbose=True,
    monitor='val_loss',
    mode='min')
    model = CAE(hparams)
    dm = hit_dm(hparams)
    trainer = Trainer.from_argparse_args(hparams,callbacks=checkpoint_callback,
                         auto_select_gpus = True,
                         precision = 16,
                         progress_bar_refresh_rate=1)  
    trainer.fit(model, datamodule = dm)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = CAE.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)