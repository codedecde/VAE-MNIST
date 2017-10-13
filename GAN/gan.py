import scipy.io as sio
from random import shuffle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from utils import Progbar
import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


WGAN = True  # The flag for setting WGAN
NORMALIZE = True
data = sio.loadmat('../mldata/mnist-original.mat')
data_x = data['data'].transpose().astype(int)
train_x = data_x[:50000]

index = range(train_x.shape[0])
shuffle(index)

# TORCH CONSTANTS
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

train_x = autograd.Variable(torch.Tensor(train_x[index]).dtype(FloatTensor))
if NORMALIZE:
    # Normalize data
    mu = train_x.mean(0)
    var = train_x.var(0) + np.finfo(float).eps
    train_x_norm = (train_x - mu) / var

val_x = data_x[50000:]
val_x = autograd.Variable(torch.Tensor(val_x).dtype(FloatTensor))
if NORMALIZE:
    val_x = (val_x - mu) / var


def plot_image(image, image_2, filename, epoch):
    image = image.data.numpy().reshape((28, 28))
    image_2 = image_2.data.numpy().reshape((28, 28))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('GAN MNIST EPOCH : %d' % epoch)
    counter = 0
    for ax in axes:
        disp_image = image if counter == 0 else image_2
        ax.imshow(disp_image, cmap='gray')
        counter += 1
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename)
    plt.close()

class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        for ix in xrange(num_layers):
            indim = self.in_dim if ix == 0 else self.hidden_dim
            outdim = self.out_dim if ix == self.num_layers - 1 else self.hidden_dim
            setattr(self, "nn_{}".format(ix), nn.Linear(indim, outdim).dtype(FloatTensor))
    def forward(self, noise):
        """
        Takes in noise and generates image
            :param noise: batch x n_in: The noise vector
            :return hidden: batch x n_out: The images generated
        """
        hidden = noise
        for ix in xrange(self.num_layers):
            hidden = getattr(self, "nn_{}".format(ix))(hidden)
            if ix != self.num_layers - 1:
                hidden = F.tanh(hidden)
        return hidden

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.nn1 = nn.Linear(in_dim, hidden_dim).dtype(FloatTensor)
        self.nn2 = nn.Linear(hidden_dim, 1).dtype(FloatTensor)
    
    def _forward(self, image):
        """
        Given image, scores it
            :param image: batch x in_dim: The collection of images
            :returns score: batch x 1 : The score of image being fake
        """
        hidden = F.tanh(self.nn1(image))
        score = self.nn2(hidden)
        return score
    
    def forward(self, fake_images, real_images):
        """
        Takes in the fake and the images and gives the difference of mean scores as loss
            :param fake_images: batch x in_dim : The fake images
            :param real_images: batch x in_dim : The real images
            :return loss: batch x 1 : The difference loss
        """
        s_fake = self._forward(fake_images)
        s_real = self._forward(real_images)
        loss = torch.mean(s_fake) - torch.mean(s_real)
        return loss


IN_DIM = 10
HIDDEN_DIM = 128

generator = Generator(IN_DIM, HIDDEN_DIM, train_x.size(1), num_layers=3)
discriminator = Discriminator(train_x.size(1), HIDDEN_DIM)

TRAIN_STEPS = 5
BATCH_SIZE = 64
PLOT_AFTER = 5
clip_norm = 0.1
lr = 1e-4
DISCRIMINATOR_THRESHOLD = 1.
curr_steps = 0.

optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
ITERATIONS = 1000000
PLOT_AFTER = 1000
batch_ix = 0


def get_numpy(array):
    return array.data.cpu().numpy()[0] if use_cuda else array.data.numpy()[0]


def zero_grad(*opts):
    for opt in opts:
        opt.zero_grad()


bar = Progbar(ITERATIONS)
for epoch in xrange(ITERATIONS):
    average_training_loss = 0.
    threshold = float(ITERATIONS - epoch) / ITERATIONS * DISCRIMINATOR_THRESHOLD
    for steps in xrange(TRAIN_STEPS):
        data_sample = train_x[batch_ix: batch_ix + BATCH_SIZE, :]
        batch_ix += BATCH_SIZE
        if batch_ix >= train_x.size(0):
            batch_ix = 0
        random_seed = autograd.Variable(torch.randn(data_sample.size(0), IN_DIM).dtype(FloatTensor))
        generator_images = generator(random_seed)
        loss_d = discriminator(generator_images, data_sample)
        average_training_loss += get_numpy(loss_d)
        loss_d.backward()
        optimizer_discriminator.step()
        for p in discriminator.parameters():
            p.data.clamp_(-.01, 0.01)
        zero_grad(optimizer_generator, optimizer_discriminator)
    average_training_loss /= TRAIN_STEPS
    # Generator
    random_seed = autograd.Variable(torch.randn(BATCH_SIZE, IN_DIM).dtype(FloatTensor))
    # loss_g = -1. * torch.sum(torch.log(discriminator._forward(generator(random_seed)))) / random_seed.size(0)
    loss_g = -torch.mean(discriminator._forward(generator(random_seed)))
    loss_g.backward()
    optimizer_generator.step()
    zero_grad(optimizer_generator, optimizer_discriminator)
    bar.update(epoch + 1, values=[("Discriminator Error", average_training_loss),
                                  ("Generator Error", get_numpy(loss_g))])
    if ((epoch + 1) % PLOT_AFTER) == 0:
        random_seed = autograd.Variable(torch.randn(BATCH_SIZE, IN_DIM).dtype(FloatTensor))
        generated_images = generator(random_seed)
        filename = "../Images_GAN/Image_Epoch_%d.png" % (epoch + 1)
        if NORMALIZE:
            # Rescale for plotting
            generated_images = (generated_images * var) + mu
        plot_image(generated_images[0], generated_images[1], filename, epoch + 1)
