import scipy.io as sio
from random import shuffle
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from keras.utils import Progbar
import pdb

data = sio.loadmat('mldata/mnist-original.mat')
data_x = data['data'].transpose().astype(int)
data_y = data['label'].transpose().astype(int)

data_x = (data_x > 128).astype(int)
train_x = data_x[:50000]
train_y = data_y[:50000]

index = range(train_x.shape[0])
shuffle(index)

train_x = autograd.Variable(torch.Tensor(train_x[index]))
train_y = autograd.Variable(torch.Tensor(train_y[index]))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()
        self.n_dim = hidden_dim
        self.in_dim = input_dim
        # Encoder
        self.encoder_nn1 = nn.Linear(self.in_dim, self.n_dim)
        self.encoder_mu = nn.Linear(self.n_dim, self.n_dim)  # mean : n_dim x 1
        self.log_sigma = nn.Linear(self.n_dim, self.n_dim)  # log_sigma : n_dim x 1
        # Decoder
        self.decoder_nn1 = nn.Linear(self.n_dim, self.n_dim)
        self.decoder_out = nn.Linear(self.n_dim, self.in_dim)

    def forward(self, x):
        """
            :params x: batch x input_dim: The input images
            :return x_new: batch x input_dim: The reconstructed image
        """
        # Encode
        e_h1 = F.relu(self.encoder_nn1(x))
        mu = self.encoder_mu(e_h1)
        log_sigma = self.log_sigma(e_h1)
        sigma = torch.exp(log_sigma / 2)
        z = mu + sigma * autograd.Variable(torch.Tensor(np.random.normal(0., 1., (x.size(0), self.n_dim))))

        # Decoder
        o_h1 = F.tanh(self.decoder_nn1(z))
        o_out = F.sigmoid(self.decoder_out(o_h1))
        return o_out, mu, log_sigma


vae = VAE(train_x.size(1), 50)
EPOCHS = 500
BATCH_SIZE = 64
optimizer = torch.optim.Adam(vae.parameters())
bce_loss = torch.nn.BCELoss()
steps = 0.
n_steps = train_x.size(0) // BATCH_SIZE if (train_x.size(0) % BATCH_SIZE) == 0 else (train_x.size(0) // BATCH_SIZE) + 1
T = EPOCHS * n_steps
for epoch in xrange(EPOCHS):
    print '\nEPOCH (%d/ %d)' % (epoch + 1, EPOCHS)
    bar = Progbar(n_steps)
    for ii, ix in enumerate(xrange(0, train_x.size(0), BATCH_SIZE)):
        batch_x = train_x[ix: ix + BATCH_SIZE]
        optimizer.zero_grad()
        new_x, mu, log_sigma = vae.forward(batch_x)
        # Compute the loss
        loss_generative = bce_loss(new_x, batch_x)
        loss_kl = torch.sum(0.5 * (torch.sum((mu * mu) + torch.exp(log_sigma) - log_sigma - 1., -1)), -1) / BATCH_SIZE
        loss = loss_generative + loss_kl.mul_(1. - np.exp(-steps / T))  # + KL divergence term
        loss.backward()
        optimizer.step()
        steps += 1.
        bar.update(ii + 1, values=[("generative_loss", loss_generative.data.numpy()[0]), ("KL_loss", loss_kl.data.numpy()[0]), ("total_loss", loss.data.numpy()[0])])
pdb.set_trace()
