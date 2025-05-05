import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader

from myDatasets import VISEMSimpleDataset
from myNetworks import LSTMNetwork

import numpy as np
import shutil


class LitNetwork(pl.LightningModule):
    def __init__(self,in_channels,out_channels,network_name="LSTM",batch_size=1,lr=1e-4,max_epochs=200,scheduler="step"):
        super(LitNetwork, self).__init__()

        if network_name == "LSTM":
            self.model = LSTMNetwork(in_channels,out_channels)
        else:
            raise ValueError("Network name not recognized")

        self.loss_func = torch.nn.MSELoss()

        #self.dropout = nn.Dropout(0.2)

        self.b = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.scheduler = scheduler

        #self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=out_channels,average='micro')

    def forward(self, x):

        x = self.model(x)

        return x

    def configure_optimizers(self):
        if self.scheduler == "step":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(.75*self.max_epochs), gamma=0.1)
        elif self.scheduler == "onecycle":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr*0.1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=1, epochs=self.max_epochs)
        else:
            raise ValueError("Scheduler type not recognized")
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, data, batch_idx):
        #video_frames, sequence = data[0], data[1]
        sequence = data

        # Store the last element of the sequence
        final_location = sequence[:, -1, :]

        # Remove the last element
        sequence = sequence[:, :-1, :]
        #video_frames = video_frames[:, -1, :]

        # Pass the sequence through the model
        out_loc = self.forward(sequence)

        loss = self.loss_func(out_loc, final_location)

        self.log("train_loss",loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.b,sync_dist=True)
        
        return loss
    
    def validation_step(self, val_data, batch_idx):
        sequence = val_data

        # Store the last element of the sequence
        final_location = sequence[:, -1, :]

        # Remove the last element
        sequence = sequence[:, :-1, :]
        #video_frames = video_frames[:, -1, :]

        # Pass the sequence through the model
        out_loc = self.forward(sequence)

        loss = self.loss_func(out_loc, final_location)

        self.log("val_loss",loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.b,sync_dist=True)
        
        return None

    def test_step(self, test_data, batch_idx):
        sequence = test_data

        # Store the last element of the sequence
        final_location = sequence[:, -1, :]

        # Remove the last element
        sequence = sequence[:, :-1, :]
        #video_frames = video_frames[:, -1, :]

        # Pass the sequence through the model
        out_loc = self.forward(sequence)

        loss = self.loss_func(out_loc, final_location)

        self.log("test_loss",loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.b,sync_dist=True)
        
        return None


def train_network(network_name="LSTM",batch_size=64,max_epochs=50,learning_rate=1e-4,scheduler="step",workers=8,progress_bar=True):

    # Load the dataset
    VISEM = VISEMSimpleDataset(root_dir="../VISEM_Simple_Dataset/")

    train_loader = DataLoader(VISEM,batch_size=batch_size,num_workers=workers,persistent_workers=True,shuffle=True)
    val_loader = DataLoader(VISEM,batch_size=batch_size,num_workers=workers,persistent_workers=True)
    test_loader = DataLoader(VISEM,batch_size=batch_size,num_workers=workers,persistent_workers=True)

    hparams = {"network_name": network_name, "batch_size": batch_size, "max_epochs": max_epochs, "learning_rate": learning_rate, "scheduler": scheduler}

    model = LitNetwork(2,2, network_name,batch_size,learning_rate,max_epochs,scheduler)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss',save_top_k=1, mode='min')
    logger = pl_loggers.TensorBoardLogger(save_dir="my_logs/", name=network_name+"_"+scheduler+"_"+str(max_epochs)+"_{lr:.0e}".format(lr=learning_rate))
    logger.log_hyperparams(hparams)
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=[checkpoint], logger=logger, enable_progress_bar=progress_bar, log_every_n_steps=10)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_loader,val_loader)
        
    trainer.test(ckpt_path="best", dataloaders=test_loader)
    
    # Save the checkpoint as a .ckpt file to the current directory
    best_model_path = checkpoint.best_model_path
    local_fn = "best_model.ckpt"

    shutil.move(best_model_path, local_fn)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Prediction Configuration')
    parser.add_argument('--network_name', type=str, help='Network name (e.g., "LSTM")', default='LSTM')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--scheduler', type=str, default="step", help='Scheduler type (default: "step")')
    parser.add_argument('--workers', type=int, default=4, help='Workers (default: 8)')
    parser.add_argument('--no_progress_bar', action="store_false", help='Remove progress bar during output (default: True)')
    args = parser.parse_args()

    network_name = args.network_name
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    learning_rate = args.learning_rate
    scheduler = args.scheduler
    workers = args.workers
    progress_bar = args.no_progress_bar

    train_network(network_name,batch_size,max_epochs,learning_rate,scheduler,workers,progress_bar)
    
    #noise_levels = [0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.75,0.99]

    #for n in noise_levels:
    #    train_network(clusters,noise_level=n,workers=8)
