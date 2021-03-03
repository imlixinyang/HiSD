# import packages
from core/utils import get_config
from core/trainer import HiSD_Trainer
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# load checkpoint
noise_dim = 32
image_size = 128
checkpoint = 'checkpoint_128_celeba-hq.pt'
trainer = HiSD_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.models.gen.load_state_dict(state_dict['gen_test'])
trainer.models.gen.cuda()

E = trainer.models.gen.encode
T = trainer.models.gen.translate
G = trainer.models.gen.decode
M = trainer.models.gen.map
F = trainer.models.gen.extract

transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
DIY your translation steps.
e.g. change both 'Bangs' (latent-guided) and 'Eyeglasses' (reference-guided) to 'with'. 
"""
steps = [
    {'type': 'latent-guided', 'tag': 0, 'attribute': 0, 'seed': None},
    {'type': 'reference-guided', 'tag': 1, 'reference': 'examples/reference_glasses_0.jpg'}
]

"""
You need to crop the image if you use your own input.
"""
input = 'examples/input_0.jpg'

"""
Do the translation and save the output.
"""
with torch.no_grad():
    x = transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda()
    c = E(x)
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
            c_trg = T(c_, s_, step['tag'])
            
    x_trg = G(c_trg)
vutils.save_image(((x_trg + 1)/ 2).data, 'examples/output.jpg', padding=0)







