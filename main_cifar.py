"""Training ResNetV1Net on CIFAR-10."""

import argparse
import os

import numpy as np
import torch  # pylint: disable=import-error
import torch.backends.cudnn as cudnn  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import torch.optim as optim  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error
import torchvision.models as models
import torchvision.transforms as transforms  # pylint: disable=import-error

from pytorch_boundaries.models.recurrence.v1net import V1Net


class ResNet18_V1Net(nn.Module):
  def __init__(self, 
               timesteps=4,
               num_classes=10,
               kernel_size=5,
               kernel_size_exc=7,
               kernel_size_inh=3,
               ):
    """ResNet with V1Net layer.
    params:
      timesteps: Int number of V1Net timesteps
      num_classes: Int number of output classes
      kernel_size: Int kernel size of V1Net input convolution.
      kernel_size_exc: Int kernel size for excitatory V1Net convolution.
      kernel_size_inh: Int kernel size for inhibitory V1Net convolution.
    example:
      >> x = torch.zeros(10, 3, 32, 32).cuda()
      >> resnet_v1net = ResNet18_V1Net(kernel_size=3,
                                       kernel_size_exc=7,
                                       kernel_size_inh=5).cuda()
      >> out = resnet_v1net(x)
    """
    super(ResNet18_V1Net, self).__init__()
    self.num_classes = num_classes
    self.timesteps = timesteps
    self.rgb_mean = np.array((0.485, 0.456, 0.406)) * 255.
    self.rgb_std = np.array((0.229, 0.224, 0.225)) * 255.
    # Convert to n, c, h, w
    self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
    self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
    self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
    self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()
    model = models.resnet18(pretrained=True).cuda()
    self.resnet_conv_1 = self.extract_layer(model, 
                                            'resnet18',
                                            'retina')
    self.v1net_conv = V1Net(64, 64, 
                            kernel_size,
                            kernel_size_exc, 
                            kernel_size_inh)
    self.resnet_conv_2 = self.extract_layer(model,
                                            'resnet18',
                                            'cortex')
    self.fc = nn.Linear(512, num_classes)

  def standardize(self, inputs):
    """Mean normalize input images."""
    # do not normalize CIFAR-10 images, normalization added to dataloader
    return inputs
  
  def forward(self, features):
    net = self.standardize(features)
    net = self.resnet_conv_1(net)
    n, c, h, w = net.shape
    net_tiled = net.repeat(self.timesteps, 1, 1, 1, 1).view(self.timesteps, n, c, h, w)
    net_tiled = torch.transpose(net_tiled, 1, 0)
    _, (net, _) = self.v1net_conv(net_tiled)
    net = self.resnet_conv_2(net)
    net = torch.flatten(net, 1)
    net = self.fc(net)
    return net

  def extract_layer(self, model, 
                    backbone_mode, 
                    key):
    if backbone_mode == 'resnet18':
      index_dict = {
          'retina': (0,4), 
          'cortex': (4,9),
      }
    start, end = index_dict[key]
    modified_model = nn.Sequential(*list(
      model.children()
      )[start:end])
    return modified_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = models.resnet18(pretrained=True)
net = ResNet18_V1Net(kernel_size=3, kernel_size_exc=7, kernel_size_inh=5)
net = net.to(device)
if device == 'cuda':
  # net = torch.nn.DataParallel(net)
  cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=5e-4,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
  print('\nEpoch: %d' % epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    if not batch_idx % 50:
      print('Iter- %s, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, 
                      train_loss/(batch_idx+1), 
                      100.*correct/total, 
                      correct, total))

def test(epoch):
  global best_acc
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

  # Save checkpoint.
  acc = 100.*correct/total
  if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    best_acc = acc


if __name__=="__main__":
  for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
