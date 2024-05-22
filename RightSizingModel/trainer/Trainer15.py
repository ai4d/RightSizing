import time
import os
import numpy as np
import wandb

import torch
import torch.nn.functional as F

from eval.test import test
from utils.logger1 import log, log_test
from Laplacian import diff_laplacian
from eval import eval_error


class Trainer15():
    def __init__(self, model, AWL, predictor, optimizer, train_loader, test_loader, output_dir, config):
        self.config = config
        self.epochs = config['epochs']
        self.model = model
        self.awl = AWL
        self.predictor = predictor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # alpha for reconstruction loss
        self.alpha = config['alpha']

        # we don't need beta-annealing any more


    def train(self, reduce='sum'):  
        # record the test loss
        test_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            self.awl.train()
                        
            # record the time for each epoch
            current_time = time.time()

            loss, loss_dict = self.train_epoch(reduce=reduce)
            
            # create a dictionary for each epoch loss
            len_dataset = len(self.train_loader.dataset)
            epoch_loss_dict = {"epoch": epoch, "epochs": self.epochs, "Loss": np.sum(loss)/len_dataset, "time": time.time() - current_time,
                               "Recon Loss": np.sum(loss_dict["Recon Loss"])/len_dataset, "KL Loss": np.sum(loss_dict["KL Loss"])/len_dataset,
                               "Laplacian Loss": np.sum(loss_dict["Laplacian Loss"])/len_dataset}
            
            log(epoch_loss_dict, self.output_dir)
            
            # Test the model after every epoch
            test_loss = test(self.model, self.test_loader, self.device)
            log_test(test_loss, self.output_dir)
            wandb.log({"Test Reconstruction Loss": test_loss})
            
            # save the model if epoch is 500, 600, 700            
            if epoch in [300, 400, 500, 600, 700, 800, 900]:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_{}.pth'.format(epoch)))
                
        # Save the model
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))

    def train_epoch(self, reduce='mean'):
        # If reduce is 'mean', the loss is averaged over all elements and batch
        # If reduce is 'sum', the loss is summed over all elements and batch

        epoch_losses = []
        
        epoch_recon_losses, epoch_kl_losses = [], []
        
        epoch_laplacian_losses = []

        
        for idx, data in enumerate(self.train_loader, 0):
            
            # we need the x to denote as the input graph
            x = data.x.to(self.device)
            
            # Feeding a batch of images into the network to obtain the output image, mu, logVar and z
            out, mu, logVar, z = self.model(x)
            z.retain_grad()

            # The loss function is KL Div + Reconstruction loss
            recon_loss = F.mse_loss(out, x, reduction=reduce)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            
            if reduce == 'mean':
                kl_divergence /= x.size(0)
            
            # compute the laplacian loss
            laplacian_loss = diff_laplacian(out)
            
            # make each loss into a list
            loss_list = [0.1 * recon_loss, 0.1 * kl_divergence, laplacian_loss]
            loss, losses = self.awl(loss_list)
        
            
            # Backpropagation based on the height loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.predictor.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            
            # record the loss: the loss is summed over all elements in a batch
            epoch_losses.append(loss.item() * x.size(0))
            epoch_recon_losses.append(0.1 * recon_loss.item() * x.size(0))
            epoch_kl_losses.append(0.1 * kl_divergence.item() * x.size(0))
            epoch_laplacian_losses.append(laplacian_loss.item())

            # using wandb to log the loss
            wandb.log({"Loss": loss.item()})
            wandb.log({"Recon Loss": 0.1 * recon_loss.item()})
            wandb.log({"KL Loss": 0.1 * kl_divergence.item()})
            wandb.log({"Laplacian Loss": laplacian_loss.item()})  
            
        # make the losses into a dictionary
        loss_dict = {"Loss": epoch_losses, "Recon Loss": epoch_recon_losses, "KL Loss": epoch_kl_losses, "Laplacian Loss": epoch_laplacian_losses} 

        return epoch_losses, loss_dict