import torch.nn.functional as F
import torch
from torch import nn
from TCN.tcn import TemporalConvNet, SimpleBlock, SimpleReverseBlock


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dims, kernel_size, dropout):
        super(VAE, self).__init__()
        self.encoder = TemporalConvNet(input_size, hidden_dims, kernel_size=kernel_size, dropout=dropout)
        self.mu =  nn.Linear(num_channels[-1], latent_dim)
        self.var =  nn.Linear(num_channels[-1], latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])


class SimpleVAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dims, kernel_size, dropout):
        super(SimpleVAE, self).__init__()
        modules = []
        num_levels = len(hidden_dims)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            modules += [SimpleBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1]*100, latent_dim)
        self.var = nn.Linear(hidden_dims[-1]*100, latent_dim)

         # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*100)

        for i in reversed(range(num_levels)):
            dilation_size = 2 ** i
            in_channels = hidden_dims[i]
            out_channels = input_size if i == 0 else hidden_dims[i-1]
            modules += [SimpleBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(result)
        log_var = self.var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 30, 100)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,*args,**kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self, num_samples, current_device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    
    def generate(self, x, **kwargs):
        return self.forward(x)[0]

# the chomp chops the padding from the right side.
# TODO: add sequence of layers with conv1d, chomp, relU, dropout 