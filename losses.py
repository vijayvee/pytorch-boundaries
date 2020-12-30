"""Helper functions to load loss functions."""
import numpy as np
import torch  # pylint:disable=import-error
import torch.nn as nn  # pylint:disable=import-error
from absl import flags
from torch.nn import functional as F  # pylint:disable=import-error

FLAGS = flags.FLAGS

def cross_entropy_loss2d(inputs, targets, loss_fun=None):
  """Compute weighted cross entropy loss."""
  n, c, h, w = inputs.size()
  weights = np.zeros((n, c, h, w))
  targets_confident = np.zeros((n, c, h, w))
  balance = FLAGS.label_lambda

  for i in range(n):
    t = targets[i, :, :, :].cpu().data.numpy()
    where_pos = t == 1.
    where_neg = t == 0.
    pos_count = where_pos.sum()
    neg_count = where_neg.sum()
    valid = pos_count + neg_count
    weights[i, where_pos] = neg_count / valid
    weights[i, where_neg] = pos_count * balance / valid
    targets_confident[i, where_pos] = 1.

  weights = torch.Tensor(weights).cuda()
  targets_confident = torch.Tensor(targets_confident)

  if loss_fun is None:
    loss_fun = F.binary_cross_entropy_with_logits
  loss = loss_fun(inputs, targets,
                  weights, size_average=False)
  return loss, targets_confident