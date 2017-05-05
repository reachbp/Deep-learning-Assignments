from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
y_dim = 10
h_dim = 128
X_dim = 4096
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def xavier_init(size):
    print("Insize Xavier ")
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)



Wzh = xavier_init(size=[opt.nz + y_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)

class _NetGC(nn.Module):
    def __init__(self):
        super(_NetGC, self).__init__()


    def forward(self, z, c):

        inputs = torch.cat([z,c], 1)
        #print("c size",c.size())
        #print("Input size",inputs.size())

        #print("Wzh size",Wzh.size())
        h = F.relu(inputs @ Wzh + bzh.repeat(inputs.size(0), 1))
        #print("Checkpoint1 from generator")
        X = F.sigmoid(h @ Whx  + bhx.repeat(h.size(0), 1))
        #print("Output from generator", X.size())
        return X

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = Variable(torch.zeros(1), requires_grad=True)

class _NetDC(nn.Module):
    def __init__(self):
        super(_NetDC, self).__init__()
    def forward(self, z, c):

        z = z.view(64, 4096)
        #print("Inside discriminator",z.size())
        inputs = torch.cat([z, c], 1)
        #print("Discriminator Input size",inputs.size())
        #print("Discriminator Wzh size",Wxh.size())
        h = F.relu(inputs @ Wxh + bxh.repeat(inputs.size(0), 1))
        y = F.sigmoid( h @  Why + bhy.repeat(h.size(0), 1))
        #print("Output from Discriminator", y.size())
        return y


netGC = _NetGC()
print(netGC)
netDC = _NetDC()
print(netDC)
criterion = nn.BCELoss()
input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
condition = torch.FloatTensor(opt.batchSize, 10)
fixed_condition = torch.FloatTensor(opt.batchSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
cnt = 0
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
criterion = nn.BCELoss()
condition = Variable(condition)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_condition = Variable(fixed_condition)
G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
ones_label = Variable(torch.ones(opt.batchSize))
zeros_label = Variable(torch.zeros(opt.batchSize))
params = G_params + D_params
# setup optimizer
optimizerG = optim.Adam(G_params, lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))

def reset_grad():
    for p in params:
        p.grad.data.zero_()
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netDC.zero_grad()
        real_cpu, y = data
        y_onehot = y.numpy()
        y_onehot = (np.arange(10) == y_onehot[:,None]).astype(np.float32)
        real_condition = torch.from_numpy(y_onehot)
        print("Size of input", real_condition.size())
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        condition.data.copy_(real_condition)
        label.data.resize_(batch_size).fill_(real_label)
        netGC.zero_grad()
        noise.data.resize_(batch_size, nz)
        noise.data.normal_(0, 1)

        G_sample = netGC(noise, condition)
        D_real = netDC(input, condition)
        D_fake = netDC(G_sample, condition)

        D_loss_real = criterion(D_real, ones_label)
        D_loss_fake = criterion(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        optimizerD.step()
        reset_grad()


        # Generator forward-loss-backward-update

        noise.data.resize_(batch_size, nz)
        noise.data.normal_(0, 1)
        
        G_sample = netGC(noise, condition)
        D_fake = netDC(G_sample, condition)
        G_loss = criterion(D_fake, ones_label)

        G_loss.backward()
        optimizerG.step()
        reset_grad()

        # Housekeeping - reset gradient
        netGC.zero_grad()

       

    if epoch % 100 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss.data.numpy(), G_loss.data.numpy()))

        c = np.zeros(shape=[opt.batchSize, y_dim], dtype='float32')
        c[:, np.random.randint(0, 10)] = 1.
        c = Variable(torch.from_numpy(c))
        samples = netGC(noise, c).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64, 64), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)


