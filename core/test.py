"""
General test script for HiSD. 
"""

# if type is 'latent-guided', you need to specifc 'attribute' and 'seed' (None means random).
# Otherwise if type is 'reference-guided', you need to specifc the 'reference' file path.
steps = [
    {'type': 'latent-guided', 'tag': 0, 'attribute': 0, 'seed': None},
    {'type': 'latent-guided', 'tag': 1, 'attribute': 0, 'seed': None},
    # {'type': 'reference-guided', 'tag': 1, 'reference': $your_reference_image}
]

from utils import get_config
from trainer import HiSD_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)

opts = parser.parse_args()

os.makedirs(opts.output_path, exist_ok=True)

config = get_config(opts.config)
noise_dim = config['noise_dim']
trainer = HiSD_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.models.gen.load_state_dict(state_dict['gen_test'])
trainer.models.gen.cuda()

E = trainer.models.gen.encode
T = trainer.models.gen.translate
G = trainer.models.gen.decode
M = trainer.models.gen.map
F = trainer.models.gen.extract

filename = time.time()
transform = transforms.Compose([transforms.Resize(config['new_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if os.path.isfile(opts.input_path):
    inputs = [opts.input_path]
else:
    inputs = [os.path.join(opts.input_path, file_name) for file_name in os.listdir(opts.input_path)]

with torch.no_grad():
    for input in inputs:
        x = transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda()
        c = E(x)
        c_trg = c
        for j in range(len(steps)):
            step = steps[j]
            if step['type'] == 'latent-guided':
                if step['seed'] is not None:
                    torch.manual_seed(step['seed'])
                    torch.cuda.manual_seed(step['seed']) 

                z = torch.randn(1, noise_dim).cuda()
                s_trg = M(z, step['tag'], step['attribute'])

            elif step['type'] == 'reference-guided':
                reference = transform(Image.open(step['reference']).convert('RGB')).unsqueeze(0).cuda()
                s_trg = F(reference, step['tag'])
            
            c_trg = T(c_trg, s_trg, step['tag'])
            
        x_trg = G(c_trg)
        vutils.save_image(((x_trg + 1)/ 2).data, os.path.join(opts.output_path, f'{os.path.basename(input)}_output.jpg'), padding=0)

