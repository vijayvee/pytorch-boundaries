"""Evaluation utility functions."""
import glob
import os

import numpy as np
import skimage.io as io  # pylint: disable=import-error
import torch  # pylint:disable=import-error
import torchvision.transforms as transforms  # pylint:disable=import-error
import torchvision.transforms.functional as F  # pylint:disable=import-error
from absl import app, flags
from PIL import Image
from scipy.io import savemat
from tqdm import tqdm

def save_image(image, prefix=None,
               path=None, curr_idx=None):
  """Write images to disk."""
  if curr_idx:
    filename = "%s_%04d.png" % (prefix.split('.')[0], 
                                curr_idx)
  else:
    filename = "%s.png" % prefix.split('.')[0]
  filename = os.path.join(path, filename)
  if image.shape[0] == 1:
    # Saves only one image
    image = image[0]
  if image.shape[0] == 1:
    image = image[0,:,:]
  image = np.uint8(image*255.)
  io.imsave(filename, image)
  return filename


def save_mat(image, prefix=None,
             path=None, curr_idx=None):
  """Write images to disk."""
  if curr_idx:
    filename = "%s_%04d.mat" % (prefix.split('.')[0], 
                                curr_idx)
  else:
    filename = "%s.mat" % prefix.split('.')[0]
  filename = os.path.join(path, filename)
  if image.shape[0] == 1:
    # Saves only one image
    image = image[0]
  if image.shape[0] == 1:
    image = image[0,:,:]
  mat_dict = {"predictions": image}
  savemat(filename, mat_dict)
  return filename


def get_test_idxs(split="test"):
  """Get image indices of test/val images."""
  path = "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/%s/" % split
  idxs = glob.glob("%s/*jpg" % path)
  idxs = [i.split("/")[-1].split(".")[0] for i in idxs]
  return idxs

def get_test_images(image_idx, split="test", cam=False):
  """Load an image from the val/test sets."""
  test_path = "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/%s/" % split
  test_img = Image.open(os.path.join(test_path, "%s.jpg"%(image_idx)))
  test_cam = os.path.join(test_path, "%s.npy" % image_idx)
  test_img = test_img.convert("RGB")
  test_img = np.array(test_img, dtype=np.float32)
  if test_img.max() > 1.:
    test_img /= 255.
  test_img *= 255.
  test_img_np = np.expand_dims(test_img, 0)
  test_img_np = np.transpose(test_img_np, (0, 3, 1, 2))
  test_img_tensor = torch.Tensor(test_img_np)
  if cam:
    test_cam = np.load(test_cam)
    test_cam = np.expand_dims(test_cam, 0)
    test_cam = np.expand_dims(test_cam, -1)
    test_cam = (test_cam - test_cam.min())/(test_cam.max() - test_cam.min())
    test_cam = np.transpose(test_cam, (0, 3, 1, 2))
    test_cam_tensor = torch.Tensor(test_cam)
    return test_img_tensor, test_img_np, test_cam_tensor
  return test_img_tensor, test_img_np