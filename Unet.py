import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms, utils
from torch import optim

import os
import numpy as np
import random
import tqdm

import time

# pode adicionar o cuda:0 para identificar qual a GPU que quer
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# usar .to(device) na parte de treino nas variáveis ct, pt e mask e no modelo também model.to(device)

class LiverDataset(Dataset):
  '''
  Liver segmentation dataset, have 3D matrix of CT, PT and mask
  
  Input: npy images
  Output: dict of tensors [channels, height, width] with CT, PT and masks
  '''

  def __init__(self, root_dir, transform=None, size=None):
    '''
    Args:
      root_dir (string): Directory with CT, PT and mask folders that have the respective images
      transform (callable, optional): Optional transform to be applied on a sample
      size (tuple, optional): Optinal resize function for 3D images
    '''

    self.root_dir = root_dir
    self.ct = os.listdir(os.path.join(root_dir, 'CT'))
    self.pt = os.listdir(os.path.join(root_dir, 'PT'))
    self.mask = os.listdir(os.path.join(root_dir, 'mask'))
    self.transform = transform
    self.size = size

  def __len__(self):
    return len(self.ct)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    ct = np.load(os.path.join(self.root_dir, 'CT', self.ct[idx])).astype('float')
    ct = np.moveaxis(ct, -1, 0) # channels first
    ct = torch.from_numpy(ct) # convert to tensor

    pt = np.load(os.path.join(self.root_dir, 'PT', self.pt[idx])).astype('float')
    pt = np.moveaxis(pt, -1, 0) # channels first
    pt = torch.from_numpy(pt) # convert to tensor

    mask = np.load(os.path.join(self.root_dir, 'mask', self.mask[idx])).astype('float')
    mask = np.moveaxis(mask, -1, 0) # channels first
    mask = torch.from_numpy(mask) # convert to tensor

    sample = {'CT': ct, 'PT': pt, 'mask': mask}

    if self.size is not None:
      for key in sample.keys():
        sample[key] = F.interpolate(sample[key].unsqueeze(0).unsqueeze(0), self.size).squeeze() # to resize a 3D image (?, B, C, W, H). Input (Tensor)


    if self.transform:
      for key in sample.keys():
        sample[key] = self.transform(sample[key])
    
    return sample

class DoubleConv(nn.Module):
  '''Composed by two (convolution => batch normalization => ReLU)'''

  def __init__(self, in_channels, out_channels, mid_channels=None):
      super().__init__()
      if not mid_channels:
          mid_channels = out_channels
      self.double_conv = nn.Sequential(
          nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(mid_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )

  def forward(self, x):
      return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
      super().__init__()
      self.maxpool_conv = nn.Sequential(
          nn.MaxPool2d(2),
          DoubleConv(in_channels, out_channels)
      )

  def forward(self, x):
      return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
      super().__init__()

      # if bilinear, use the normal convolutions to reduce the number of channels
      if bilinear:
          self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
          self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
      else:
          self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
          self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
      x1 = self.up(x1)
      # input is CHW
      diffY = x2.size()[2] - x1.size()[2]
      diffX = x2.size()[3] - x1.size()[3]

      x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
      # if you have padding issues, see
      # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
      # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
      x = torch.cat([x2, x1], dim=1)
      return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
      super(OutConv, self).__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
      return self.conv(x)

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss

def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    dice = float(2 * (gt * seg).sum())/float(gt.sum() + seg.sum())
    return dice

class UNet(nn.Module):
  def __init__(self, n_channels, n_classes, bilinear=False):
      super(UNet, self).__init__()
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.bilinear = bilinear

      self.inc = DoubleConv(n_channels, 64)
      self.down1 = Down(64, 128)
      self.down2 = Down(128, 256)
      self.down3 = Down(256, 512)
      factor = 2 if bilinear else 1
      self.down4 = Down(512, 1024 // factor)
      self.up1 = Up(1024, 512 // factor, bilinear)
      self.up2 = Up(512, 256 // factor, bilinear)
      self.up3 = Up(256, 128 // factor, bilinear)
      self.up4 = Up(128, 64, bilinear)
      self.outc = OutConv(64, n_classes)

  def forward(self, x):
      x1 = self.inc(x)
      x2 = self.down1(x1)
      x3 = self.down2(x2)
      x4 = self.down3(x3)
      x5 = self.down4(x4)
      x = self.up1(x5, x4)
      x = self.up2(x, x3)
      x = self.up3(x, x2)
      x = self.up4(x, x1)
      logits = self.outc(x)
      return logits

# Create dataset
dataset = LiverDataset(root_dir='/A/luismoura/projects/liver_segmentation/data/npy', size=(28, 128, 128))
# Split dataset in train/test
train_data, test_data = random_split(dataset, [16, 3])
# Create one dataloader for each split
train_dataloader = DataLoader(train_data, batch_size = 1, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = True)

# Define Model
model = UNet(n_channels=28, n_classes=28).to(device)
# Define Loss
cost = BinaryDiceLoss()
# Define Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 50
for e in range(epochs):
  print('Epoch: ', e+1)
  runing_loss = 0
  for sample in tqdm.tqdm(train_dataloader):
    #Flatten
    optimizer.zero_grad()

    ct = sample['CT'].type(torch.cuda.FloatTensor).to(device)

    output = model.forward(ct)

    mask = sample['mask'].type(torch.LongTensor).to(device)

    loss = cost(output, mask)
    # evaluate the cost function
    output = output.squeeze().data.cpu().numpy()
    label = sample['mask'].squeeze().cpu().numpy()
    dice = dice_coeff(output, label)

    optimizer.zero_grad()                                                              # we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
    loss.backward()
    optimizer.step()

    runing_loss += loss.item()
  else:
    print(f"Training loss: {runing_loss/len(train_dataloader)}")

