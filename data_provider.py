"""Helper function for data provider."""
import glob
import os

import numpy as np
import torch  # pylint:disable=import-error
import torchvision.transforms as transforms  # pylint:disable=import-error
import torchvision.transforms.functional as F  # pylint:disable=import-error
from PIL import Image

_IMAGENET_MEAN = np.array([])

def load_img(img_path):
  img = np.array(Image.open(img_path))
  if img.shape[-1] == 1:
    img = np.repeat(img, 3, axis=-1)
  img = img / 255.
  img = np.expand_dims(img, 0)
  return img


def get_train_val_ids():
  """Get image ids for train and val."""
  train_ids = glob.glob(
    "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/train/*jpg")
  val_ids = glob.glob(
    "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/val/*jpg")
  train_ids = [i.split("/")[-1].split(".")[0] for i in train_ids]
  val_ids = [i.split("/")[-1].split(".")[0] for i in val_ids]
  return {"train": train_ids, "val": val_ids}


class BSDSDataProvider(object):
  """Data provider for BSDS500."""
  def __init__(self,
               image_size,
               is_training,
               data_dir,
               ):
    self.image_size = image_size
    self.is_training = is_training
    self.data_dir = data_dir
    self.img_gt_paths = []

    if self.is_training:
      self.data_file = os.path.join(self.data_dir,
                                     "train_pair.lst")
      self.img_ids = get_train_val_ids()["train"]
    else:
      self.data_file = os.path.join(self.data_dir,
                                    "train_pair.lst")
      self.img_ids = get_train_val_ids()["val"]

    with open(self.data_file, "r") as f:
      all_files = f.read()

    all_files = all_files.strip().split("\n")
    for f in all_files:
      img, gt = f.split(" ")
      img_id = img.split("/")[-1].split(".")[0]
      if img_id in self.img_ids:
        self.img_gt_paths.append((img, gt))
    self.num_samples = len(self.img_gt_paths)

  def transform(self, images, labels):
    """Transform images and ground truth."""
    if self.is_training:
      seed = np.random.randint(2**32) # make a seed with numpy generator 
      np.random.seed(seed) # apply this seed to img tranfsorms
      torch.manual_seed(seed)
      resize_transform_img = transforms.RandomResizedCrop(self.image_size, 
                                                      scale=(0.75, 1.25),
                                                      ratio=(0.9, 1.1))
      images = resize_transform_img(images)

      np.random.seed(seed) # apply this seed to target tranfsorms
      torch.manual_seed(seed) # needed for torchvision 0.7
      resize_transform_gt = transforms.RandomResizedCrop(self.image_size, 
                                                      scale=(0.75, 1.25),
                                                      ratio=(0.9, 1.1))
      labels = resize_transform_gt(labels)
      
      color_transform = transforms.ColorJitter(brightness=0.3,
                                               contrast=0.3,
                                               saturation=0.3,
                                               hue=0.1
                                               )
      images = color_transform(images)
    images = np.array(images)
    labels = np.array(labels)
    # Renormalize labels to [0, 1]
    labels = (labels - labels.min())/(labels.max() - labels.min())
    images = F.to_tensor(images)
    labels = F.to_tensor(labels)
    return images, labels

  def __getitem__(self, idx):
    img, gt = self.img_gt_paths[idx]
    img = os.path.join(self.data_dir, img)
    gt = os.path.join(self.data_dir, gt)
    img = Image.open(img).convert("RGB")
    gt = Image.open(gt).convert("L")
    img, gt = self.transform(img, gt)
    return img, gt

  def __len__(self):
    return self.num_samples
