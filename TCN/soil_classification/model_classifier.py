from numpy.lib import index_tricks
import torch.nn.functional as F
import torch
from torch import nn


class LatentClassifier(nn.Module):
    def __init__(self, back_model, n_features, hidden_dims, n_classes, VAE = False, dropout=0.2):
        super(LatentClassifier, self).__init__()
        self.back_model = back_model
        self.VAE = VAE
        in_dim = n_features
        modules = []
        for dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_dim,dim), nn.Sigmoid(), nn.Dropout(dropout)
            ))
            in_dim = dim
        self.linear = nn.Sequential(*modules)
        self.output = nn.Linear(in_dim, n_classes)
    
    def freeze_back(self):
        for para in self.back_model.parameters():
            para.requires_grad = False

    def forward(self, inputs):
        latent_features = self.back_model.encode(inputs)
        if self.VAE:
            mu, var = latent_features[0], latent_features[1]
            latent_features = self.back_model.reparameterize(mu, var)
        #print(latent_features.shape)
        latent_features = torch.flatten(latent_features, start_dim=1)
        out = self.linear(latent_features)
        out = self.output(out)
        return F.log_softmax(out, dim=1)


