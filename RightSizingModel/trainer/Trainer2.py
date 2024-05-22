"""
Height + waist

We tried two different Loss functions:
1. loss = alpha * recon_loss + beta * (kl_divergence + 100 * height_loss + 100 * waist_loss)    ==> This one is better
   20231206-180205
   It can disentangle the height and waist better. Only the second latent dimension controls the waist change.
   But the reconstruction result is not good. 
2. loss = alpha * recon_loss + beta * kl_divergence + 100 * height_loss + 100 * waist_loss
    20231205-222033
   It can reconstruct the meshes better. But it looks the last dimension is also controlling the waist change, but not that obvious.
"""

import time
import os
import numpy as np
import wandb

import torch
import torch.nn.functional as F

from eval.test import test
from utils.logger1 import log, log_test

class Trainer2():
    def __init__(self, model, predictor, optimizer, train_loader, test_loader, output_dir, config):
        self.config = config
        self.epochs = config['epochs']
        self.model = model
        self.predictor = predictor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # alpha for reconstruction loss
        self.alpha = config['alpha']

        # beta annealing
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.beta_anneal_epochs = config['beta_anneal_epochs']


    def train(self, reduce='mean'):  
        # record the test loss
        test_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            
            # record the time for each epoch
            current_time = time.time()

            # beta annealing
            current_bata = min(self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.beta_anneal_epochs), self.beta_end)

            loss, loss_dict = self.train_epoch(alpha=self.alpha, beta=current_bata, reduce=reduce)
            
            # create a dictionary for each epoch loss
            len_dataset = len(self.train_loader.dataset)
            epoch_loss_dict = {"epoch": epoch, "epochs": self.epochs, "Loss": np.sum(loss)/len_dataset, "time": time.time() - current_time,
                               "Recon Loss": np.sum(loss_dict["Recon Loss"])/len_dataset, "KL Loss": np.sum(loss_dict["KL Loss"])/len_dataset,
                               "Height Loss": np.sum(loss_dict["Height Loss"])/len_dataset, "Waist Loss": np.sum(loss_dict["Waist Loss"])/len_dataset}
            
            log(epoch_loss_dict, self.output_dir)
            
            # Test the model after every epoch
            test_loss = test(self.model, self.test_loader, self.device)
            log_test(test_loss, self.output_dir)
            wandb.log({"Test Reconstruction Loss": test_loss})

        # Save the model
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))

    def train_epoch(self, alpha, beta, reduce='mean'):
        # If reduce is 'mean', the loss is averaged over all elements and batch
        # If reduce is 'sum', the loss is summed over all elements and batch

        epoch_losses = []
        
        epoch_recon_losses, epoch_kl_losses = [], []
        epoch_height_losses, epoch_waist_loss = [], []
        
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

            # construct the canonical matrix
            # z is of shape (batch_size, latent_dim)

            # Now, we need to compute the attributes of the mesh
            # 0: height, 1: arm_length, 2: crotch_height, 
            # 3: chest_circumference, 4: hip_circumference, 5: waist_circumference,
            predicted_result = self.predictor(out)
            height = predicted_result[:, 0]
            waist = predicted_result[:, 5]

            # Compute the gradient of height/waist w.r.t. z
            height_grad = torch.autograd.grad(height, z, grad_outputs=torch.ones(height.size()).cuda(), retain_graph=True, create_graph=True)[0]
            waist_grad = torch.autograd.grad(waist, z, grad_outputs=torch.ones(waist.size()).cuda(), retain_graph=True, create_graph=True)[0]
            
            # compute the the grad loss and divide by the batch size
            height_loss = torch.norm(height_grad[:, 1:], p=2) 
            waist_loss = torch.norm(torch.concat((waist_grad[:, 0:1], waist_grad[:, 2:]), dim=1), p=2) 
            
            loss = alpha * recon_loss + beta * (kl_divergence + 100 * height_loss + 100 * waist_loss)      

            # Backpropagation based on the height loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.predictor.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            
            # record the loss: the loss is summed over all elements in a batch
            epoch_losses.append(loss.item() * x.size(0))
            epoch_recon_losses.append(alpha * recon_loss.item() * x.size(0))
            epoch_kl_losses.append(beta * kl_divergence.item() * x.size(0))
            epoch_height_losses.append(beta * 100 * height_loss.item() )
            epoch_waist_loss.append(beta * 100 * waist_loss.item())
            
            # using wandb to log the loss
            wandb.log({"Loss": loss.item()})
            wandb.log({"Recon Loss": alpha * recon_loss.item()})
            wandb.log({"KL Loss": beta * kl_divergence.item()})
            wandb.log({"Height Loss": 100 * height_loss.item()})
            wandb.log({"Waist Loss": 100 * waist_loss.item()})   
            
        # make the losses into a dictionary
        loss_dict = {"Loss": epoch_losses, "Recon Loss": epoch_recon_losses, "KL Loss": epoch_kl_losses, 
                     "Height Loss": epoch_height_losses, "Waist Loss": epoch_waist_loss} 

        return epoch_losses, loss_dict