import torch.nn.functional as F
import torch
from torch import nn
from TCN.tcn import TemporalConvNet, TemporalConvNetRev


class TemplateTCNAE(nn.Module):
    """
    A template for the temporal convolutional network autoencoder. Provides the temporal networks and loss function.
    Child classes should define what occurs between this.
    """
    def __init__(self, input_size, hidden_dims, kernel_size, dropout):
        super(TemplateTCNAE, self).__init__()
        self.encoder_tcn = TemporalConvNet(input_size, hidden_dims, kernel_size=kernel_size, dropout=dropout)
        self.decoder_tcn = TemporalConvNetRev(input_size, hidden_dims, kernel_size=kernel_size, dropout=dropout)
        #self.decoder_tcn = TemporalConvNet(input_size, hidden_dims, kernel_size=kernel_size, dropout=dropout)
    
    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        features = args[2]
        loss = F.mse_loss(recons, input)
        return loss


class TCNAE(nn.Module):
    def __init__(self, input_size, latent_dim, pool_size,  hidden_dims, kernel_size, dropout):
        super(TCNAE, self).__init__()
        self.encoder_tcn = TemporalConvNet(input_size, hidden_dims, kernel_size=kernel_size, dropout=dropout)
        self.encoder_conv = nn.Conv1d(hidden_dims[-1], latent_dim, kernel_size=1)
        self.encoder_pool = nn.AvgPool1d(pool_size)
        self.encoder = nn.Sequential(self.encoder_tcn, self.encoder_conv, self.encoder_pool)

        self.decoder_upsample = nn.Upsample(scale_factor=pool_size)
        self.decoder_tcn = TemporalConvNet(latent_dim, hidden_dims, kernel_size=kernel_size, dropout=dropout)
        self.decoder_conv = nn.Conv1d(hidden_dims[-1], input_size, kernel_size=1)
        self.decoder = nn.Sequential(self.decoder_upsample, self.decoder_tcn, self.decoder_conv)

    def encode(self, input):
        result = self.encoder(input)
        return result
    
    def decode(self, features):
        result = self.decoder(features)
        return result
    
    def forward(self, input, **kwargs):
        features = self.encoder(input)
        reconstruction = self.decoder(features)
        return [reconstruction, input, features]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        features = args[2]
        loss = F.l1_loss(recons, input)
        return loss

class TCNAERevDialation(TCNAE):
    def __init__(self, input_size, latent_dim, pool_size,  hidden_dims, kernel_size, dropout):
        super(TCNAERevDialation, self).__init__(input_size, latent_dim, pool_size,  hidden_dims, kernel_size, dropout)
        self.decoder_tcn = TemporalConvNetRev(latent_dim, hidden_dims, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Sequential(self.decoder_upsample, self.decoder_tcn, self.decoder_conv)

class DenseTCNAE(TemplateTCNAE):
    def __init__(self, input_size, latent_dim, seq_len,  hidden_dims, kernel_size, dropout):
        super(DenseTCNAE, self).__init__(input_size, hidden_dims, kernel_size, dropout)
        self.seq_len = seq_len
        self.last_hidden_dim = hidden_dims[-1]
        print(self.last_hidden_dim)

        self.latent_features = nn.Linear(self.last_hidden_dim*self.seq_len, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.last_hidden_dim*self.seq_len)
        
    def encode(self, input):
        result = self.encoder_tcn(input)
        print(result)
        result = torch.flatten(result, start_dim=1)
        latent_features = self.latent_features(result)
        return latent_features

    def decode(self, lf):
        result = self.decoder_input(lf)
        result = result.view(-1, self.last_hidden_dim, self.seq_len)
        recons = self.decoder_tcn(result)
        return recons

    def forward(self, input, **kwargs):
        latent_features = self.encode(input)
        recons = self.decode(latent_features)
        return [recons, input, latent_features]

class ConvTCNAE(TemplateTCNAE):
    def __init__(self, input_size, scale, hidden_dims, kernel_size, dropout):
        super(ConvTCNAE, self).__init__(input_size, hidden_dims, kernel_size, dropout)
        self.last_hidden_dim = hidden_dims[-1]
        self.latent_features = nn.Conv1d(self.last_hidden_dim, 5, kernel_size=scale, stride=scale, padding_mode='replicate')
        #self.decoder_upsample = nn.Upsample(scale_factor=scale)
        #self.decoder_input = nn.Conv1d(1, self.last_hidden_dim, kernel_size=1, stride=1)
        #self.decoder_input = nn.Sequential(self.decoder_upsample, self.decoder_input)
        self.decoder_input = nn.ConvTranspose1d(5, self.last_hidden_dim, kernel_size=scale, stride=scale)

    def encode(self, input):
        result = self.encoder_tcn(input)
        latent_features = self.latent_features(result)
        return latent_features

    def decode(self, lf):
        result = self.decoder_input(lf)
        print(result.shape)
        recons = self.decoder_tcn(result)
        return recons
    
    def forward(self, input, **kwargs):
        latent_features = self.encode(input)
        recons = self.decode(latent_features)
        return [recons, input, latent_features]

class DenseLastTCNAE(TemplateTCNAE):
    def __init__(self, input_size, latent_dim, seq_len,  hidden_dims, kernel_size, dropout):
        super(DenseLastTCNAE, self).__init__(input_size, hidden_dims, kernel_size, dropout)
        self.seq_len = seq_len
        self.last_hidden_dim = hidden_dims[-1]
        print(self.last_hidden_dim)

        self.latent_features = nn.Linear(self.last_hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.last_hidden_dim*self.seq_len)
        
    def encode(self, input):
        result = self.encoder_tcn(input)
        print(result)
        latent_features = self.latent_features(result[:,:,-1])
        return latent_features

    def decode(self, lf):
        result = self.decoder_input(lf)
        result = result.view(-1, self.last_hidden_dim, self.seq_len)
        recons = self.decoder_tcn(result)
        return recons

    def forward(self, input, **kwargs):
        latent_features = self.encode(input)
        recons = self.decode(latent_features)
        return [recons, input, latent_features]


class DenseAE(nn.Module):
    def __init__(self, input_channels, seq_len, latent_dim, hidden_dims):
        super(DenseAE, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        modules = []
        in_dim = input_channels*seq_len
        for dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_dim,dim), nn.ReLU()))
            in_dim = dim
        self.encoder = nn.Sequential(*modules)
        self.latent_features = nn.Linear(in_dim,latent_dim)

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
        latent_features = self.latent_features(result)
        return latent_features

    def decode(self, lf):
        result = self.decoder_input(lf)
        result = self.decoder(result)
        recons = self.output_layer(result)
        recons = recons.reshape((-1,self.input_channels,self.seq_len))
        return recons

    def forward(self, input, **kwargs):
        latent_features = self.encode(input)
        recons = self.decode(latent_features)
        return [recons, input, latent_features]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        features = args[2]
        loss = F.l1_loss(recons, input)
        return loss
            
class ConvAE(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size):
        super(ConvAE, self).__init__()
        modules = []
        in_filters = input_channels 
        for out_filters in num_filters:
            modules.append(nn.Sequential(nn.Conv1d(in_filters,out_filters, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2)), nn.ReLU()))
            in_filters = out_filters

        self.encoder = nn.Sequential(*modules)

        # maybe later add something linear here

        modules = []
        num_filters.reverse()
        for i in range(len(num_filters)-1):
            modules.append(nn.Sequential(nn.ConvTranspose1d(num_filters[i],num_filters[i+1], kernel_size=kernel_size,stride=2, padding=int((kernel_size-1)/2), output_padding=1), nn.ReLU()))
        modules.append(nn.ConvTranspose1d(num_filters[-1],input_channels, kernel_size=kernel_size,stride=2, padding=int((kernel_size-1)/2), output_padding=1))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        latent_features = self.encoder(input)
        #print(latent_features.shape)
        return latent_features

    def decode(self, lf):
        recons = self.decoder(lf)
        return recons

    def forward(self, input, **kwargs):
        latent_features = self.encode(input)
        recons = self.decode(latent_features)
        return [recons, input, latent_features]

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        features = args[2]
        loss = F.l1_loss(recons, input)
        return loss


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
