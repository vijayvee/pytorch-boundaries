"""Evaluating BSDS trained models."""
import glob
import os

import numpy as np
import torch  # pylint:disable=import-error
import torchvision.transforms as transforms  # pylint:disable=import-error
import torchvision.transforms.functional as F  # pylint:disable=import-error
from absl import flags
from PIL import Image
from scipy.io import savemat

from pytorch_boundaries.data_provider import BSDSDataProvider
from pytorch_boundaries.models.vgg16_config import vgg16_hed_config
from pytorch_boundaries.models.vgg_16_hed import VGG_HED

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpointdir", "",
                    "Checkpoint file path")

flags.DEFINE_string("outdir", "",
                    "Output file path")

outputs = ["side_output_1",
           "side_output_2",
           "side_output_3",
           "side_output_4",
           "side_output_5",
           "fused_output",
           "all_outputs",
           "gated_output",
           "gated_fusion",
           ]


def get_model():
  """Load VGG-HED model."""
  cfg = vgg16_hed_config("vgg16_bn", 400, 1, False, False)
  model = VGG_HED(cfg)
  return model


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
  if image.shape[-1] == 1:
    image = image[:,:,0]
  mat_dict = {"predictions": image}
  savemat(filename, mat_dict)
  return filename


def get_test_idxs(split="test"):
  """Get image indices of test/val images."""
  path = "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/%s/" % split
  idxs = glob.glob("%s/*jpg" % path)
  idxs = [i.split("/")[-1].split(".")[0] for i in idxs]
  return idxs


def get_test_images(image_idx, split="test"):
  """Load an image from the val/test sets."""
  test_path = "/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/%s/" % split
  test_img = Image.open(os.path.join(test_path, "%s.jpg"%(image_idx)))
  test_img = test_img.convert("RGB")
  test_img = np.array(test_img, dtype=np.float32)
  if test_img.max() > 1.:
    test_img /= 255.
  test_img *= 255.
  test_img_np = np.expand_dims(test_img, 0)
  test_img_np = np.transpose(test_img_np, (0, 3, 1, 2))
  test_img_tensor = torch.Tensor(test_img_np)
  return test_img_tensor, test_img_np


def evaluate():
  """Evaluation loop."""
  d_outdir = {}
  for output in outputs:
    outdir = os.path.join(FLAGS.outdir, output)
    os.mkdir(outdir)
    d_outdir[output] = outdir

  l_checkpoints = glob.glob("%s/*pth" % FLAGS.checkpointdir)
  for checkpoint in l_checkpoints:
    with torch.no_grad():
      model = get_model()
      model.to("cuda")
      state_dict = torch.load(checkpoint)
      model.load_state_dict(state_dict)
      model.eval()
      test_imgs = get_test_idxs()
      for img in test_imgs:
        tch_img, _ = get_test_images(img)
        preds = model(tch_img)
        nppreds = {k: torch.sigmoid(v).cpu().numpy() 
                      for k, v in preds.items()}
        side_outputs = [nppreds["side_output_%s" % i]
                          for i in range(1, 6)]
        y = side_outputs[-1]
        side_outputs = side_outputs[:-1]
        for x in side_outputs[::-1]:
          y = np.multiply(x, y)
        nppreds["gated_output"] = y
        nppreds["all_outputs"] = np.array([
                                v for k, v in nppreds.items() 
                                if "output" in k])
        nppreds["all_outputs"] = np.mean(nppreds["all_outputs"], 0)
        nppreds["gated_fusion"] = (nppreds["fused_output"] \
                                  + nppreds["gated_output"])/2
        for output, nppred in nppreds.items():
          save_mat(nppred, img, d_outdir[output])




