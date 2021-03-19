"""
The main codes are form MUNIT
"""
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageAttributeDataset
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time

def get_data_iters(conf, gpus):
    batch_size = conf['batch_size']
    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    num_workers = conf['num_workers']
    tags  = conf['tags']

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list 
    transform_list = [transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)] + transform_list 
    transform = transforms.Compose(transform_list)

    loaders = [[DataLoader(
        dataset=ImageAttributeDataset(tags[i]['attributes'][j]['filename'], transform),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
                            for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    iters = [[data_prefetcher(loader, batch_size, gpus) for loader in loaders] for loaders in loaders]

    return iters

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [torch.clamp(images, -1, 1).expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n], display_image_num, '%s/gen_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.01)

    return init_fun

class data_prefetcher():
    def __init__(self, loader, batch_size, gpus):
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.batch_size = batch_size
        self.gpu0 = int(gpus[0])

        self.preload()

    def preload(self):
        try:
            self.x, self.y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            self.x, self.y = next(self.iter)

        if self.x.size(0) != self.batch_size:
            self.iter = iter(self.loader)
            self.x, self.y = next(self.iter)
        
        with torch.cuda.stream(self.stream):
            self.x, self.y = self.x.cuda(self.gpu0, non_blocking=True), self.y.cuda(self.gpu0, non_blocking=True)

    def next(self):
        return self.x, self.y