"""Common model utilities."""

import numpy as np
import torch  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error


def crop_tensor(net, inputs):
  """Crop net to input height and width."""
  _, _, in_h, in_w = net.shape
  _, _, out_h, out_w = inputs.shape
  assert in_h >= out_h and in_w >= out_w
  x_offset = (in_w - out_w) // 2
  y_offset = (in_h - out_h) // 2
  if x_offset or y_offset:
    cropped_net = net[:, :, y_offset:y_offset+out_h, x_offset:x_offset+out_w]
  return cropped_net

def get_upsampling_weight(in_channels=1, out_channels=1, kernel_size=4):
  """Make a 2D bilinear kernel suitable for upsampling"""
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  og = np.ogrid[:kernel_size, :kernel_size]
  filt = (1 - abs(og[0] - center) / factor) * \
          (1 - abs(og[1] - center) / factor)
  weight = np.zeros((in_channels, out_channels, 
                     kernel_size, kernel_size),
                    dtype=np.float32)
  weight[range(in_channels), range(out_channels), :, :] = filt
  weight = torch.from_numpy(weight).float()
  weight = weight.cuda()
  return weight
