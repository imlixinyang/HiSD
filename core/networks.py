from torch import nn
import torch
import torch.nn.functional as F

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
from utils import weights_init
import math

##################################################################################
# Discriminator
##################################################################################

class Dis(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters['tags']
        channels = hyperparameters['discriminators']['channels']

        self.conv = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels[-1] + 
            # ALI part which is not shown in the original submission but help disentangle the extracted style. 
            hyperparameters['style_dim'] +
            # Tag-irrelevant part. Sec.3.4
            self.tags[i]['tag_irrelevant_conditions_dim'],
            # One for translated, one for cycle. Eq.4
            len(self.tags[i]['attributes'] * 2), 1, 1, 0),
        ) for i in range(len(self.tags))])

    def forward(self, x, s, y, i):
        f = self.conv(x)
        fsy = torch.cat([f, tile_like(s, f), tile_like(y, f)], 1)
        return self.fcs[i](fsy).view(f.size(0), 2, -1)
        
    def calc_dis_loss_real(self, x, s, y, i, j):
        loss = 0
        x = x.requires_grad_()
        out = self.forward(x, s, y, i)[:, :, j]
        loss += F.relu(1 - out[:, 0]).mean()
        loss += F.relu(1 - out[:, 1]).mean()
        loss += self.compute_grad2(out[:, 0], x)
        loss += self.compute_grad2(out[:, 1], x)
        return loss
    
    def calc_dis_loss_fake_trg(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = F.relu(1 + out[:, 0]).mean()
        return loss
    
    def calc_dis_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = F.relu(1 + out[:, 1]).mean()
        return loss

    def calc_gen_loss_real(self, x, s, y, i, j):
        loss = 0
        out = self.forward(x, s, y, i)[:, :, j]
        loss += out[:, 0].mean()
        loss += out[:, 1].mean()
        return loss

    def calc_gen_loss_fake_trg(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = - out[:, 0].mean()
        return loss

    def calc_gen_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = - out[:, 1].mean()
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
         )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg.mean()

##################################################################################
# Generator
##################################################################################

class Gen(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters['tags']

        self.style_dim = hyperparameters['style_dim']
        self.noise_dim = hyperparameters['noise_dim']

        channels = hyperparameters['encoder']['channels']
        self.encoder = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[EncoderBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )    

        channels = hyperparameters['decoder']['channels']
        self.decoder = nn.Sequential(
            *[GeneratorBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Conv2d(channels[-1], hyperparameters['input_dim'], 1, 1, 0)
        )   

        self.extractors = Extractors(hyperparameters)

        self.translators = nn.ModuleList([Translator(hyperparameters)
            for i in range(len(self.tags))]
        )
        
        self.mappers =  nn.ModuleList([Mapper(hyperparameters, len(self.tags[i]['attributes']))
            for i in range(len(self.tags))]
        )

    def encode(self, x):
        e = self.encoder(x)
        return e

    def decode(self, e):
        x = self.decoder(e)
        return x

    def extract(self, x, i):
        return self.extractors(x, i)
    
    def map(self, z, i, j):
        return self.mappers[i](z, j)

    def translate(self, e, s, i):
        return self.translators[i](e, s)


##################################################################################
# Extractors, Translator and Mapper
##################################################################################

class Extractors(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.num_tags = len(hyperparameters['tags'])
        channels = hyperparameters['extractors']['channels']
        self.model = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DiscriminatorBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1],  hyperparameters['style_dim'] * self.num_tags, 1, 1, 0),
        )

    def forward(self, x, i):
        s = self.model(x).view(x.size(0), self.num_tags, -1)
        return s[:, i]

class Translator(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        channels = hyperparameters['translators']['channels']
        self.model = nn.Sequential( 
            Conv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[TranslatorBlock(channels[i], channels[i + 1], hyperparameters['style_dim']) for i in range(len(channels) - 1)]
        )
        
        self.features = nn.Sequential(
            Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0, style_dim=hyperparameters['style_dim']),
        ) 

        self.masks = nn.Sequential(
            Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0, style_dim=hyperparameters['style_dim']),
            
            nn.Sigmoid()
        ) 
    
    def forward(self, e, s):
        mid = self.model(e)
        f = self.features(mid)
        m = self.masks(mid) 

        return f * m + e * (1 - m)


class Mapper(nn.Module):
    def __init__(self, hyperparameters, num_attributes):
        super().__init__()
        channels = hyperparameters['mappers']['pre_channels']
        self.pre_model = nn.Sequential(
            MapperBlock(hyperparameters['noise_dim'], channels[0]),
            *[MapperBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hyperparameters['mappers']['post_channels']
        self.post_models = nn.ModuleList([nn.Sequential(
            *[MapperBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            Linear(channels[-1], hyperparameters['style_dim']), 
            ) for i in range(num_attributes)
        ])

    def forward(self, z, j):
        z = self.pre_model(z)
        return self.post_models[j](z)

##################################################################################
# Basic Blocks
##################################################################################

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.model = Residual(
            nn.Sequential(
                Conv2d(in_dim, out_dim, demod=False, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),
                Conv2d(in_dim, out_dim, demod=False, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.AvgPool2d(2),
            )
        )

    def forward(self, x):
        return self.model(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.model = Residual(
            nn.Sequential(
                Conv2d(in_dim, in_dim, demod=True, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),
                Conv2d(in_dim, out_dim, demod=True, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.AvgPool2d(2),
            )
        )

    def forward(self, x):
        return self.model(x)

class TranslatorBlock(nn.Module):
    def __init__(self, in_dim, out_dim, style_dim):
        super().__init__()
        
        self.model = Residual(
            nn.Sequential(
                Conv2d(in_dim, out_dim, style_dim=style_dim, demod=True, add_noise=True, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
                Conv2d(out_dim, out_dim, style_dim=style_dim, demod=True, add_noise=True, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Identity()
            )
        )

    def forward(self, x):
        return self.model(x)

class GeneratorBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.model = Residual(
            nn.Sequential(
                Conv2d(in_dim, out_dim, demod=True, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Unsample2d(scale_factor=2, mode='bilinear'),
                Conv2d(out_dim, out_dim, demod=True, scale=math.sqrt(2/(1+0.2**2))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Unsample2d(scale_factor=2, mode='bilinear'),
            )
        )

    def forward(self, x):
        return self.model(x)

class MapperBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = Linear(in_dim, out_dim, scale=math.sqrt(2))
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))

##################################################################################
# Basic Modules and Functions
##################################################################################

class Residual(nn.Module):
    def __init__(self, forward_model, skip_model, scale=1/math.sqrt(2)):
        super().__init__()
        self.forward_model = forward_model
        self.skip_model = skip_model

        self.scale = scale

    def forward(self, x):
        return (self.skip_model(x) + self.forward_model(x)) * self.scale


class Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1,
                 scale=1, style_dim=None, demod=False, add_noise=False, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, kernel_size, kernel_size) / math.sqrt(in_dim * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        self.stride = stride
        self.padding= padding
        self.scale = scale
        self.demod = demod
        self.eps = 1e-8

        if style_dim is not None:
            self.mod = True
            self.style = None
            self.style_to_mod = nn.Linear(style_dim, in_dim)
        else:
            self.mod = False

        if add_noise:
            self.noise_weight = nn.Parameter(torch.zeros(out_dim)) 
        else:
            self.noise_weight = None

    def forward(self, x):
        B, C_in, H, W = x.shape

        if self.mod:
            assert self.style is not None
            modulation = self.style_to_mod(self.style).add_(1)
            weight = self.weight[None] * modulation[:, None, :, None, None]

            if self.demod:
                weight = weight / torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)

            weight = weight.reshape(-1, *weight.shape[2:])
            x = x.reshape(1, -1, H, W)

            y = F.conv2d(x, weight * self.scale, stride=self.stride, padding=self.padding, groups=B)
            y = y.view(B, -1, *y.shape[2:])

        else:
            weight = self.weight
            if self.demod:
                weight = weight / torch.rsqrt((weight ** 2).sum(dim=(1, 2, 3), keepdim=True) + self.eps)

            weight = weight
            y = F.conv2d(x, self.weight * self.scale, stride=self.stride, padding=self.padding)

        y = y.add_(self.bias[None, :, None, None]) if self.bias is not None else y
        y = y.add_(self.noise_weight[None, :, None, None] * torch.randn_like(y[:, :1])) if self.noise_weight is not None else y

        return y

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, scale=1, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        self.scale = scale
    
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, bias=self.bias)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def tile_like(x, target):
    # make x is able to concat with target at dim 1.
    x = x.view(x.size(0), -1, 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x