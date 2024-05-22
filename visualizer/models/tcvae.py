import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

import math
from typing import List, Any

from models.conv import ChebConv

def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class Enblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Enblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, down_transform):
        out = F.elu(self.conv(x, edge_index))
        out = Pool(out, down_transform)
        return out


class Deblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Deblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out, edge_index))
        return out


class BetaTCVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels, out_channels, latent_channels, edge_index,
                 down_transform, up_transform, K,
                 anneal_steps: int = 200,
                 alpha: float = 1.,
                 beta: float = 6.,
                 gamma: float = 1.,
                 **kwargs) -> None:
        super(BetaTCVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_vert = self.down_transform[-1].size(0)
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Build encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels, out_channels[idx], K, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(out_channels[idx - 1], out_channels[idx], K,
                            **kwargs))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        # The following is to add variational part into account
        # Linear layer to generate mu and sigma
        self.mu_linear = nn.Linear(latent_channels, latent_channels)
        self.sigma_linear = nn.Linear(latent_channels, latent_channels)

        # Build Decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K,
                            **kwargs))
            else:
                self.de_layers.append(
                    Deblock(out_channels[-idx], out_channels[-idx - 1], K,
                            **kwargs))
        # reconstruction
        self.de_layers.append(
            ChebConv(out_channels[0], in_channels, K, **kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.edge_index[i], self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)

        # get the mu/sigma value
        mu = self.mu_linear(x)
        log_var = torch.exp(self.sigma_linear(x))

        return [mu, log_var]

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.edge_index[num_deblocks - i],
                          self.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, self.edge_index[0])
        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var, z]

    def log_density_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at which Gaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self, recons, x, mu, log_var, z, dataset_size) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param recons: the output of the decoder
        :param x: the input of the encoder
        :param mu/log_var: the output of the encoder
        :param z: sampled from mu and log_var
        :param dataset_size: size of training dataset
        :param args:
        :param kwargs:
        :return:
        """

        # kwargs['M_N']
        # Account for the minibatch samples from the dataset
        weight = 1

        recons_loss = F.mse_loss(recons, x, reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size, latent_dim),
                                                log_var.view(1, batch_size, latent_dim))

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(x.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss / batch_size + self.alpha * mi_loss + weight * (
                    self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss,
                'TC_Loss': tc_loss,
                'MI_Loss': mi_loss}



    # def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map. This function is for evaluation results.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(num_samples, self.latent_dim)
    #
    #     z = z.to(current_device)
    #
    #     samples = self.decode(z)
    #     return samples
    #
    # def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """
    #
    #     return self.forward(x)[0]