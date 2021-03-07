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
            *[DownBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )    

        channels = hyperparameters['decoder']['channels']
        self.decoder = nn.Sequential(
            *[UpBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
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
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
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
            nn.Conv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[MiddleBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.style_to_params = nn.Linear(hyperparameters['style_dim'], self.get_num_adain_params(self.model))
        
        self.features = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
        ) 

        self.masks = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
            nn.Sigmoid()
        ) 
    
    def forward(self, e, s):
        p = self.style_to_params(s)
        self.assign_adain_params(p, self.model)

        mid = self.model(e)
        f = self.features(mid)
        m = self.masks(mid) 

        return f * m + e * (1 - m)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params


class Mapper(nn.Module):
    def __init__(self, hyperparameters, num_attributes):
        super().__init__()
        channels = hyperparameters['mappers']['pre_channels']
        self.pre_model = nn.Sequential(
            nn.Linear(hyperparameters['noise_dim'], channels[0]),
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hyperparameters['mappers']['post_channels']
        self.post_models = nn.ModuleList([nn.Sequential(
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Linear(channels[-1], hyperparameters['style_dim']), 
            ) for i in range(num_attributes)
        ])

    def forward(self, z, j):
        z = self.pre_model(z)
        return self.post_models[j](z)

##################################################################################
# Basic Blocks
##################################################################################

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)

class DownBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(in_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(self.in2(F.avg_pool2d(self.conv1(self.activ(self.in1(x.clone()))), 2))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.conv1(F.interpolate(self.activ(x.clone()), scale_factor=2, mode='nearest'))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.in2(self.conv1(F.interpolate(self.activ(self.in1(x.clone())), scale_factor=2, mode='nearest')))))
        out = residual + out
        return out / math.sqrt(2)

class MiddleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.adain2(self.conv1(self.activ(self.adain1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))

##################################################################################
# Basic Modules and Functions
##################################################################################

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.bias = None
        self.weight = None

    def forward(self, x):
        assert self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def tile_like(x, target):
    # make x is able to concat with target at dim 1.
    x = x.view(x.size(0), -1, 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x