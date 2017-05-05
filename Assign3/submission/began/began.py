import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torchvision import datasets, transforms


dataset = datasets.ImageFolder(root='pkmn',
                               transform=transforms.Compose([
                                   transforms.Scale(64),
                                   transforms.ToTensor()
                               ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=int(4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=0, padding=0, ceil_mode=True)

        # Starting image is of size 64, 4 times maxpooled down to 8.
        # Map to the embedding size.
        self.fc_to_embedding = nn.Linear(64 * 8 * 8, 128)
        
        # Map the embedding to an 8x8 image with 16 channels
        self.fc_to_image = nn.Linear(128, 8 * 8 * 16)

        self.conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_f = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # Activation function
        self.activation = F.relu
        self.activation = lambda x: x

    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        x = self.activation(self.maxpool(self.conv1(x)))
        x = self.activation(self.maxpool(self.conv2(x)))
        x = self.activation(self.maxpool(self.conv3(x)))
        x = self.activation(self.conv4(x))

        # Flatten the image
        x = x.view(-1, 64 * 8 * 8)

        x = self.activation(self.fc_to_embedding(x))
        
        x = self.activation(self.fc_to_image(x))
        # Convert to an image
        x = x.view(-1, 16, 8, 8)
        
        # Apply convolutions and upsample the image
        x = self.activation(self.upsample(self.conv(self.conv(x))))
        x = self.activation(self.upsample(self.conv(self.conv(x))))
        x = self.activation(self.upsample(self.conv(self.conv(x))))
        
        x = self.activation(self.conv_f(x))
        
        # Reconvert to 1D
        x = x.view(-1, 3 * 64 * 64)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Map the embedding to an 8x8 image with 16 channels
        self.fc1 = nn.Linear(128, 8 * 8 * 16)

        self.conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_f = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        
        # Activation function
        self.activation = F.relu
        self.activation = lambda x: x
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        # Convert to an image
        x = x.view(-1, 16, 8, 8)
        
        # Apply convolutions and upsample the image
        x = self.activation(self.upsample(self.conv(self.conv(x))))
        x = self.activation(self.upsample(self.conv(self.conv(x))))
        x = self.activation(self.upsample(self.conv(self.conv(x))))
        
        x = self.activation(self.conv_f(x))
        
        # Reconvert to 1D
        x = x.view(-1, 3 * 64 * 64)
        return x

# D is an autoencoder, approximating Gaussian
def D(X):
    X_recon = D_(X)
    # Use Laplace MLE as in the paper
    return torch.mean(torch.sum(torch.abs(X - X_recon), 1))

def reset_grad():
    G.zero_grad()
    D_.zero_grad()

G = Generator()
D_ = Discriminator()

batch_size = 16
embedding_size = 128
height = 64
width = 64
channels = 3
learning_rate = 0.0002

cuda = False

cnt = 0
d_step = 3
m = 5
lam = 1e-3
k = 0
gamma = 0.5

if (cuda):
  G.cuda()
  D_.cuda()

G_solver = optim.Adam(G.parameters(), lr=learning_rate)
D_solver = optim.Adam(D_.parameters(), lr=learning_rate)

# Train
epochs = 100
for i in range(epochs):
  for batch_idx, (data, target) in enumerate(dataloader):
      # Sample data
      data = data.view(-1, height * width * channels)
      X = Variable(data)

      if cuda:
        X = X.cuda()

      # Dicriminator
      # Create a random tensor of size [mini_batch x embedding_vector_dim]
      z_D = Variable(torch.randn(batch_size, embedding_size))

      if cuda:
        z_D = z_D.cuda()

      # Define discriminator loss
      D_loss = D(X) - k * D(G(z_D))

      D_loss.backward()
      D_solver.step()
      reset_grad()

      # Generator
      z_G = Variable(torch.randn(batch_size, embedding_size))

      if cuda:
        z_G = z_G.cuda()

      G_loss = D(G(z_G))

      G_loss.backward()
      G_solver.step()
      reset_grad()

      # Update k, the equlibrium
      k = k + lam * (gamma*D(X) - D(G(z_G)))
      k = k.data[0]

      # Print and plot every now and then
      if i % 2 == 0:
          measure = D(X) + torch.abs(gamma*D(X) - D(G(z_G)))

          print('Iter-{}; Convergence measure: {:.4}'
                .format(batch_idx, measure.data[0]))

          samples = G(z_G).data.numpy()[:16]

          fig = plt.figure(figsize=(4, 4))
          gs = gridspec.GridSpec(4, 4)
          gs.update(wspace=0.05, hspace=0.05)

          for i, sample in enumerate(samples):
              ax = plt.subplot(gs[i])
              plt.axis('off')
              ax.set_xticklabels([])
              ax.set_yticklabels([])
              ax.set_aspect('equal')
              plt.imshow(sample.reshape(3, 64, 64).transpose(1, 2, 0))

          if not os.path.exists('out/'):
              os.makedirs('out/')

          plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
          cnt += 1
          plt.close(fig)
