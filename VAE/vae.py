import scipy.io as sio
from random import shuffle, choice
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


BINARY = True


data = sio.loadmat('../mldata/mnist-original.mat')
data_x = data['data'].transpose().astype(int)
data_x = (data_x > 128).astype(int) if BINARY else data_x

train_x = data_x[:50000]
index = range(train_x.shape[0])
shuffle(index)
train_x = autograd.Variable(torch.Tensor(train_x[index]))

val_x = data_x[50000:]
val_x = autograd.Variable(torch.Tensor(val_x))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.n_dim = hidden_dim
        self.in_dim = input_dim
        self.latent_dim = latent_dim
        # Encoder
        self.encoder_nn1 = nn.Linear(self.in_dim, self.n_dim)
        self.encoder_mu = nn.Linear(self.n_dim, self.latent_dim)  # mean : latent_dim x 1
        self.encoder_logvar = nn.Linear(self.n_dim, self.latent_dim)  # logvar : latent_dim x 1
        # Decoder
        self.decoder_nn1 = nn.Linear(self.latent_dim, self.n_dim)
        self.decoder_out = nn.Linear(self.n_dim, self.in_dim)

    def encode(self, x):
        """
        Encodes the input into latent mu and sigma
            :params x: batch x n_in: the input image
            :returns mu: batch x latent_dim: The conditional mean
            :returns logvar: batch x latent_dim: The conditional log variance values of diag matrix
        """
        e_h = F.tanh(self.encoder_nn1(x))
        mu = self.encoder_mu(e_h)
        logvar = self.encoder_logvar(e_h)
        return mu, logvar

    def decode(self, z):
        """
        Decodes the latent variable to reconstruct image
            :param z: batch x latent_dim: The latent variable
            :returns out: The reconstructed image
        """
        o_h = F.tanh(self.decoder_nn1(z))
        out = F.sigmoid(self.decoder_out(o_h)) if BINARY else self.decoder_out(o_h)
        return out

    def forward(self, x):
        """
            :params x: batch x input_dim: The input images
            :return x_new: batch x input_dim: The reconstructed image
        """
        # Encode
        mu, logvar = self.encode(x)
        sigma = torch.exp(logvar / 2)
        random_seed = np.random.multivariate_normal(np.zeros((sigma.size(1),)), np.eye(sigma.size(1), sigma.size(1)), x.size(0))
        z = mu + sigma * autograd.Variable(torch.Tensor(random_seed))
        # Decode
        out = self.decode(z)
        return out, mu, logvar

    def generate(self, image=None):
        """
        Generates sample using random seed if image is not given, else conditioned on the image
            :param image: 1 x input_dim: The image as generated
        """
        random_seed = np.random.multivariate_normal(np.zeros((self.latent_dim,)), np.eye(self.latent_dim, self.latent_dim), 1)
        random_seed = autograd.Variable(torch.Tensor(random_seed))
        if image is not None:
            mu, logvar = self.encode(image)
            sigma = torch.exp(logvar / 2)
            random_seed = mu + sigma * random_seed
        # Now use the random seed to generate the image
        return self.decode(random_seed)


def plot_image(new_image, original_image, filename, epoch):
    new_image = new_image.data.numpy().reshape((28, 28))
    new_image = (new_image > 0.5).astype(int) if BINARY else new_image
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


vae = VAE(train_x.size(1), 400, 20)
EPOCHS = 100
BATCH_SIZE = 100
GENERATE_AFTER = 5
optimizer = torch.optim.Adagrad(vae.parameters(), lr=0.01)
loss_function = torch.nn.BCELoss() if BINARY else torch.nn.MSELoss()
steps = 0.
n_steps = train_x.size(0) // BATCH_SIZE if (train_x.size(0) % BATCH_SIZE) == 0 else (train_x.size(0) // BATCH_SIZE) + 1
T = EPOCHS * n_steps
for epoch in xrange(EPOCHS):
    print '\nEPOCH (%d/ %d)' % (epoch + 1, EPOCHS)
    bar = Progbar(n_steps)
    for ii, ix in enumerate(xrange(0, train_x.size(0), BATCH_SIZE)):
        batch_x = train_x[ix: ix + BATCH_SIZE]
        optimizer.zero_grad()
        new_x, mu, logvar = vae.forward(batch_x)
        # Compute the loss
        loss_generative = loss_function(new_x, batch_x)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalizing by the number of reconstructions seems clutch.
        # Doesn't work without it for binary. Gives better results for gaussian.
        # Am not really sure why though ...
        loss_kl /= BATCH_SIZE * 784
        loss = loss_generative + loss_kl
        loss.backward()
        optimizer.step()
        steps += 1.
        bar.update(ii + 1, values=[("generative_loss", loss_generative.data.numpy()[0]), ("KL_loss", loss_kl.data.numpy()[0]), ("total_loss", loss.data.numpy()[0])])
    if (epoch + 1) % GENERATE_AFTER == 0:
        original_image = choice(val_x).view(1, -1)
        image = vae.generate(original_image)
        image_filename = "Images/Image_epoch%d.png" % (epoch + 1) if BINARY else "Images/Image_Gaussian_epoch%d.png" % (epoch + 1)
        plot_image(image, original_image, image_filename, epoch + 1)
