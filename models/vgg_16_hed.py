"""VGG16-HED implementation."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error
import torchvision.models as models  # pylint: disable=import-error

from pytorch_boundaries.utils.model_utils import crop_tensor, get_upsampling_weight

def conv_weights_init(m):
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, 
                                mode='fan_out', 
                                nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class VGG_HED(nn.Module):
  def __init__(self, config):
    super(VGG_HED, self).__init__()
    self.num_classes = config.num_classes
    self.rgb_mean = np.array((0.485, 0.456, 0.406))
    self.rgb_std = np.array((0.229, 0.224, 0.225))
    # Convert to n, c, h, w
    self.rgb_mean = self.rgb_mean.reshape((1, 3, 1, 1))
    self.rgb_mean = torch.Tensor(self.rgb_mean).float().cuda()
    self.rgb_std = self.rgb_std.reshape((1, 3, 1, 1))
    self.rgb_std = torch.Tensor(self.rgb_std).float().cuda()

    model = models.vgg16(pretrained=True).cuda()
    self.conv_1 = self.extract_layer(model, "vgg16", 1)
    self.conv_2 = self.extract_layer(model, "vgg16", 2)
    self.conv_3 = self.extract_layer(model, "vgg16", 3)
    self.conv_4 = self.extract_layer(model, "vgg16", 4)
    self.conv_5 = self.extract_layer(model, "vgg16", 5)

    self.dsn1 = nn.Conv2d(64, 1, 1)
    self.dsn2 = nn.Conv2d(128, 1, 1)
    self.dsn3 = nn.Conv2d(256, 1, 1)
    self.dsn4 = nn.Conv2d(512, 1, 1)
    self.dsn5 = nn.Conv2d(512, 1, 1)

    self.dsn1_bn = nn.BatchNorm2d(1)
    self.dsn2_bn = nn.BatchNorm2d(1)
    self.dsn3_bn = nn.BatchNorm2d(1)
    self.dsn4_bn = nn.BatchNorm2d(1)
    self.dsn5_bn = nn.BatchNorm2d(1)

    self.dsn2_up = nn.ConvTranspose2d(1, 1, 4,
                                      stride=2)
    self.dsn3_up = nn.ConvTranspose2d(1, 1, 8,
                                      stride=4)
    self.dsn4_up = nn.ConvTranspose2d(1, 1, 16,
                                      stride=8)
    self.dsn5_up = nn.ConvTranspose2d(1, 1, 32,
                                      stride=16)

    self.fuse_conv = nn.Conv2d(5, 1, 1)

    init_conv_layers = [self.dsn1, self.dsn2, self.dsn3, 
                        self.dsn4, self.dsn5,
                        self.dsn1_bn, self.dsn2_bn, self.dsn3_bn, 
                        self.dsn4_bn, self.dsn5_bn,
                        ]

    for layer in init_conv_layers:
      layer.apply(conv_weights_init)

    self.fuse_conv.weight.data.fill_(0.2)
    self.fuse_conv.bias.data.fill_(0) 
    self.upconv_weights_init()

  def upconv_weights_init(self):
    """Initialize the transpose convolutions."""
    for name, m in self.named_modules():
      if isinstance(m, nn.ConvTranspose2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        dsn_idx = int(name.split("dsn")[-1].split("_")[0])
        kernel_size = 2**dsn_idx
        m.weight.copy_(get_upsampling_weight(1, 1, kernel_size))
        nn.init.constant_(m.bias, 0)
      
  def standardize(self, inputs):
    """Mean normalize input images."""
    inputs = inputs - self.rgb_mean
    inputs = inputs / self.rgb_std
    return inputs
  
  def forward(self, inputs):
    net = self.standardize(inputs)
    net = self.conv_1(net)
    self.side_output_1 = self.dsn1_bn(
                            self.dsn1(net))
    net = self.conv_2(net)
    self.side_output_2 = crop_tensor(
                          self.dsn2_up(
                            self.dsn2_bn(
                              self.dsn2(net))),
                          inputs)
    net = self.conv_3(net)
    self.side_output_3 = crop_tensor(
                          self.dsn3_up(
                            self.dsn3_bn(
                              self.dsn3(net))),
                          inputs)
    net = self.conv_4(net)
    self.side_output_4 = crop_tensor(
                          self.dsn4_up(
                            self.dsn4_bn(
                              self.dsn4(net))),
                          inputs)
    net = self.conv_5(net)
    self.side_output_5 = crop_tensor(
                          self.dsn5_up(
                            self.dsn5_bn(
                              self.dsn5(net))),
                          inputs)
    stacked_outputs = torch.cat((self.side_output_1,
                                 self.side_output_2,
                                 self.side_output_3,
                                 self.side_output_4,
                                 self.side_output_5,),
                                 dim=1)
    net = self.fuse_conv(stacked_outputs)
    return_dict = {"fused_output": net,
                   "side_output_1": self.side_output_1,
                   "side_output_2": self.side_output_2,
                   "side_output_3": self.side_output_3,
                   "side_output_4": self.side_output_4,
                   "side_output_5": self.side_output_5,
                   }
    return return_dict
      
  def extract_layer(self, model, backbone_mode, ind):
    if backbone_mode=='vgg16':
        index_dict = {
            1: (0,4), 
            2: (4,9), 
            3: (9,16), 
            4: (16,23),
            5: (23,30) }
    elif backbone_mode=='vgg16_bn':
        index_dict = {
            1: (0,6), 
            2: (6,13), 
            3: (13,23), 
            4: (23,33),
            5: (33,43) }

    start, end = index_dict[ind]
    modified_model = nn.Sequential(*list(
      model.features.children()
      )[start:end])
    return modified_model