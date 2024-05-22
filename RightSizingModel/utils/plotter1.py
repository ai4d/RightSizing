import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import torchvision
import torch


# plot the loss curve:
def plot_loss_curve(loss: List, output_dir, title, filename):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# plot the recon loss curve and kl loss curve
def plot_recon_kl_curve(recon_loss: List, kl_loss: List, output_dir):
    plt.figure()
    plt.plot(recon_loss, label='Reconstruction Loss')
    plt.plot(kl_loss, label='KL Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss and KL Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'recon_kl_curve.pdf'))
    plt.close()
