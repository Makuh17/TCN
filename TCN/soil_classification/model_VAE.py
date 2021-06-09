import torch.nn.functional as F
import torch
from torch import nn
from TCN.tcn import TemporalConvNet, TemporalConvNetRev

class DenseVAE(nn.Module):
    def __init__(self, input_channels, seq_len, latent_dim, hidden_dims):
        super(DenseVAE, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        modules = []
        in_dim = input_channels*seq_len
        for dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_dim,dim), nn.ReLU()))
            in_dim = dim
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(in_dim,latent_dim)
        self.logvar = nn.Linear(in_dim,latent_dim)

        self.decoder_input = nn.Linear(latent_dim,hidden_dims[-1])
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(nn.Linear(hidden_dims[i],hidden_dims[i+1]), nn.ReLU()))
        self.decoder = nn.Sequential(*modules)
        self.output_layer = nn.Linear(hidden_dims[-1],input_channels*seq_len)

    def encode(self, input):
        input = torch.flatten(input, start_dim=1)
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(result)
        log_var = self.logvar(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.output_layer(result)
        result = result.reshape((-1,self.input_channels,self.seq_len))
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.l1_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
            
class ConvVAE(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size):
        super(ConvVAE, self).__init__()
        modules = []
        in_filters = input_channels 
        for out_filters in num_filters:
            modules.append(nn.Sequential(nn.Conv1d(in_filters,out_filters, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2)), nn.ReLU()))
            in_filters = out_filters

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Conv1d(in_filters,in_filters,kernel_size=1, stride=1)
        self.logvar = nn.Conv1d(in_filters,in_filters,kernel_size=1, stride=1)
        #self.mu = nn.Linear()

        # maybe later add something linear here

        modules = []
        num_filters.reverse()
        for i in range(len(num_filters)-1):
            modules.append(nn.Sequential(nn.ConvTranspose1d(num_filters[i],num_filters[i+1], kernel_size=kernel_size,stride=2, padding=int((kernel_size-1)/2), output_padding=1), nn.ReLU()))
        modules.append(nn.ConvTranspose1d(num_filters[-1],input_channels, kernel_size=kernel_size,stride=2, padding=int((kernel_size-1)/2), output_padding=1))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(result)
        log_var = self.logvar(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        #print(z.shape)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = torch.flatten(args[2], start_dim=1)
        log_var = torch.flatten(args[3], start_dim=1)

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.l1_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}


#TODO:
#   Construct a proper testing scheme
#       report the test loss during training
#       somehow plot the reconstructed data 
#   Replicate model from  http://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
#       Just simply copy the TCN architecture and follow it by conv and pooling
#       For the decoder do the same, but have upsampling before and conv after
#   Use same model as above, but with a dense layer instead of pooling
#   
#   Try reducing to two or three features to plot it and label based on class
