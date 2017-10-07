import scipy.io as sio
from random import shuffle, random, choice
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from utils import Progbar
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data = sio.loadmat('mldata/mnist-original.mat')
data_x = data['data'].transpose().astype(int)
data_x = (data_x > 128).astype(int)

train_x = data_x[:50000]
index = range(train_x.shape[0])
shuffle(index)
train_x = autograd.Variable(torch.Tensor(train_x[index]))

val_x = data_x[50000:]
val_x = autograd.Variable(torch.Tensor(val_x))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()
        self.n_dim = hidden_dim
        self.in_dim = input_dim
        # Encoder
        self.encoder_nn1 = nn.Linear(self.in_dim, self.n_dim)
        self.encoder_mu = nn.Linear(self.n_dim, self.n_dim)  # mean : n_dim x 1
        self.encoder_log_sigma_square = nn.Linear(self.n_dim, self.n_dim)  # log_sigma_square : n_dim x 1
        # Decoder
        self.decoder_nn1 = nn.Linear(self.n_dim, self.n_dim)
        self.decoder_out = nn.Linear(self.n_dim, self.in_dim)

    def forward(self, x):
        """
            :params x: batch x input_dim: The input images
            :return x_new: batch x input_dim: The reconstructed image
        """
        # Encode
        e_h1 = F.tanh(self.encoder_nn1(x))
        mu = self.encoder_mu(e_h1)
        log_sigma_square = self.encoder_log_sigma_square(e_h1)
        sigma = torch.exp(log_sigma_square / 2)
        random_seed = np.random.multivariate_normal(np.zeros((self.n_dim,)), np.eye(self.n_dim, self.n_dim), x.size(0))
        z = mu + sigma * autograd.Variable(torch.Tensor(random_seed))

        # Decoder
        o_h1 = F.tanh(self.decoder_nn1(z))
        o_out = F.sigmoid(self.decoder_out(o_h1))
        return o_out, mu, log_sigma_square

    def generate(self, image=None):
        """
            Generates sample using random seed if image is not given, else conditioned on the image
            :param image: 1 x input_dim: The image as generated
        """
        random_seed = np.random.multivariate_normal(np.zeros((self.n_dim,)), np.eye(self.n_dim, self.n_dim), 1)
        random_seed = autograd.Variable(torch.Tensor(random_seed))
        if image is not None:
            e_h1 = F.relu(self.encoder_nn1(image))
            mu = self.encoder_mu(e_h1)
            log_sigma_square = self.encoder_log_sigma_square(e_h1)
            sigma = torch.exp(log_sigma_square / 2)
            random_seed = mu + sigma * random_seed
        # Now use the random seed to generate the image
        o_h1 = F.tanh(self.decoder_nn1(random_seed))
        o_out = F.sigmoid(self.decoder_out(o_h1))
        return o_out


def plot_image(new_image, original_image, filename, epoch):
    new_image = new_image.data.numpy().reshape((28, 28))
    new_image = (new_image > 0.5).astype(int)
    original_image = original_image.data.numpy().reshape((28, 28)) if original_image is not None else np.zeros((28, 28))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('VAE MNIST EPOCH : %d' % epoch)
    counter = 0
    for ax in axes:
        disp_image = new_image if counter == 0 else original_image
        ax.imshow(disp_image, cmap='gray')
        counter += 1
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename)


vae = VAE(train_x.size(1), 500)
EPOCHS = 500
BATCH_SIZE = 100
GENERATE_AFTER = 10
optimizer = torch.optim.Adagrad(vae.parameters(), lr=0.01)
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
        new_x, mu, log_sigma_square = vae.forward(batch_x)
        # Compute the loss
        loss_generative = bce_loss(new_x, batch_x)
        # loss_kl = torch.sum(0.5 * (torch.sum((mu * mu) + torch.exp(log_sigma) - log_sigma - 1., -1)), -1) / BATCH_SIZE
        loss_kl = torch.sum(0.5 * (torch.sum((mu * mu) + torch.exp(log_sigma_square) - log_sigma_square - 1., -1)), -1) / BATCH_SIZE
        loss = loss_generative + loss_kl.mul_(1. - np.exp(-steps / T))
        # loss = loss_generative + loss_kl
        loss.backward()
        optimizer.step()
        steps += 1.
        bar.update(ii + 1, values=[("generative_loss", loss_generative.data.numpy()[0]), ("KL_loss", loss_kl.data.numpy()[0]), ("total_loss", loss.data.numpy()[0])])
    if (epoch + 1) % GENERATE_AFTER == 0:
        if random() < 0.5:
            original_image = choice(val_x).view(1, -1)
        else:
            original_image = None
        image = vae.generate(original_image)
        plot_image(image, original_image, "Images/Image_epoch%d.png" % (epoch + 1), epoch + 1)
