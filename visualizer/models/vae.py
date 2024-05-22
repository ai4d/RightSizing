

from torch import nn
import torch.nn.functional as F
import torch


class AEBlock(nn.Module):
    def __init__(self, in_channel, out_channel, leaky_relu_slope=0.2):
        super(AEBlock, self).__init__()
        self.linear_layer = nn.Linear(in_channel, out_channel)
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x):
        out = self.linear_layer(x)
        out = F.leaky_relu(input=out, negative_slope=self.leaky_relu_slope)
        return out


class VariationalEncoder(nn.Module):
    def __init__(self, in_channel, out_channels, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.in_channel = in_channel
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(AEBlock(in_channel=in_channel, out_channel=out_channels[0]))
            else:
                self.en_layers.append(AEBlock(in_channel=out_channels[idx-1], out_channel=out_channels[idx]))

        # Linear layer to generate mu and sigma
        self.mu_linear = nn.Linear(out_channels[-1], latent_dims)
        self.sigma_linear = nn.Linear(out_channels[-1], latent_dims)

        self.kl = 0

    def reparameterize(self, mu, logvar):
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

    def forward(self, x):
        for i, layer in enumerate(self.en_layers):
            x = layer(x)

        # get the mu/sigma value
        mu = self.mu_linear(x)
        log_var = self.sigma_linear(x)

        # re-parameterization trick
        z = self.reparameterize(mu=mu, logvar=log_var)

        # compute the kl divergence
        self.kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return z


class VariationalDecoder(nn.Module):
    def __init__(self, in_channel, out_channels, latent_dims):
        super(VariationalDecoder, self).__init__()

        self.de_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(AEBlock(in_channel=latent_dims, out_channel=out_channels[0]))
            else:
                self.de_layers.append(AEBlock(in_channel=out_channels[idx-1], out_channel=out_channels[idx]))

        self.linear_layer = nn.Linear(out_channels[idx], in_channel)

    def forward(self, x):
        for i, layer in enumerate(self.de_layers):
            x = layer(x)
        x_hat = self.linear_layer(x)
        # x_hat = torch.sigmoid(input=x)

        return x_hat


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dims, in_channel, encoder_out_channels, decoder_out_channels):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VariationalEncoder(in_channel=in_channel, out_channels=encoder_out_channels, latent_dims=latent_dims)
        self.decoder = VariationalDecoder(in_channel=in_channel, out_channels=decoder_out_channels, latent_dims=latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

