"""Training ResNetV1Net on CIFAR-10."""

import argparse
import os
import sys

import numpy as np
import torch  # pylint: disable=import-error
import torch.backends.cudnn as cudnn  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import torch.optim as optim  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error
from absl import app, flags

from pytorch_boundaries.models.resnet_v1net import ResNet18_V1Net
from pytorch_boundaries.models.preact_resnet import PreActResNet18, PreActResNet50, PreActResNetV1Net18

remove_v1net = True
test_run = False
timesteps = 0
write_results = None  # "cifar_predictions/resnet18_v1net_predictions_eval_t_%s_remove_v1net_%s" % (timesteps, remove_v1net)
checkpoint = None  # checkpoint/ckpt_4steps_reg_remove_v1net_True.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = models.resnet18(pretrained=True)
# net = ResNet18_V1Net(kernel_size=3, kernel_size_exc=7, 
#                      kernel_size_inh=5, timesteps=timesteps,
#                      remove_v1net=remove_v1net)
net = PreActResNetV1Net18(num_classes=100, kernel_size=3, 
                          kernel_size_exc=5, kernel_size_inh=3, 
                          timesteps=3)
# net = PreActResNet18(num_classes=100)
net = net.to(device)
if device == 'cuda':
  # net = torch.nn.DataParallel(net)
  cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2,
                       momentum=0.9, nesterov=True,
                       weight_decay=1e-3)
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
  if checkpoint:
    state_dict = torch.load(checkpoint)
    print("Loading from %s with accuracy %s" % (checkpoint, 
                                                state_dict["acc"]))
    net.load_state_dict(state_dict['net'], strict=False)
  test_loss = 0
  correct = 0
  total = 0
  np_predictions = []
  np_targets = []

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      if write_results:
        np_predictions.extend(outputs.cpu().numpy())
        np_targets.extend(targets.cpu().numpy())
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if write_results:
      np.array(np_predictions).dump(
            open('%s_acc_%.3f_predictions.npy' % (write_results, acc), 'wb'))
      np.array(np_targets).dump(
            open('%s_acc_%.3f_labels.npy' % (write_results, acc), 'wb'))
      return

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
    torch.save(state,
      './checkpoint/cifar100_ckpt_0steps_reg_remove_v1net_%s_run2.pth' % remove_v1net)
    best_acc = acc


def main():
  if test_run:
    test(-1)
  else:
    for epoch in range(start_epoch, start_epoch+200):
      train(epoch)
      test(epoch)
      if epoch > 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
          param_group['lr'] /= 2.

if __name__=="__main__":
  main()