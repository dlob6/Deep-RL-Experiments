from collections import namedtuple
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torchvision.transforms as T
import fastai
from fastai.text.models import MultiHeadAttention, PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 'rectified' relu - ensuring mean(RReLU(x)) = 0
# assuming mean(x) = 0. Not the case with ReLU(x)
class RReLU(nn.Module):
    def __init__(self, residual = False):
        super().__init__()
        self.rectifier = - 1 / np.sqrt(2 * np.pi)
        self.act = nn.ReLU()
        
    def forward(self, x):
        return self.act(x) + self.rectifier
    
class RLReLU(nn.Module):
    def __init__(self, slope = 0.01):
        super().__init__()
        self.slope = slope
        self.rectifier = - (1 - slope) / np.sqrt(2 * np.pi)
        self.act = nn.LeakyReLU(negative_slope = slope, inplace = True)
        
    def forward(self, x):
        return self.act(x) + self.rectifier

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_screen(env, size = 80):
    
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').mean(axis=2)[...,None].transpose((2, 0, 1)) 
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape

    #screen = screen[:, int(screen_height*0.3):int(screen_height*0.9), int(screen_width*0.25):int(screen_width*0.75)]
    screen = screen[:, int(screen_height*0.1):int(screen_height*0.95),:]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = (np.ascontiguousarray(screen, dtype=np.float32) / 255.0)
    screen = torch.from_numpy(screen)
    if size is None: return screen.unsqueeze(0).to(device)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((size, size), interpolation=Image.CUBIC),
                        T.ToTensor()])
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


class Transformer(nn.Module):
    """Transformer model - code from fastai. Removed normalization layer and input embeddings"""
    def __init__(self, ctx_len:int, n_layers:int, n_heads:int, d_model:int, 
                 d_head:int, d_inner:int,
                 act = RLReLU(), resid_p:float=0, attn_p:float=0, 
                 ff_p:float=0, embed_p:float=0, bias:bool=True, 
                 scale:bool=True, double_drop:bool=False, 
                 attn_cls = MultiHeadAttention, mask:bool=False):
        super().__init__()
        
        self.mask = mask
        self.pos_enc = nn.Embedding(ctx_len, d_model)
        self.drop_emb = nn.Dropout(embed_p)
        # 'n_layers' Transformer layers. 
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner,
                                                  act=act, resid_p=resid_p, attn_p=attn_p,
                                                  ff_p=ff_p, bias=bias, scale=scale, 
                                                  double_drop=double_drop, attn_cls=attn_cls) 
                                     for k in range(n_layers)])

    def reset(self): pass
    def select_hidden(self, idxs): pass

    def forward(self, x):
        bs, x_len, emb_size = x.size()
        
        pos = torch.arange(0, x_len, device=x.device, dtype = torch.int64)
        pos = self.pos_enc(pos)[None]
        inp = self.drop_emb(x + pos)
        mask = torch.triu(x.new_ones(x_len, x_len), diagonal=1).bool()[None,None] if self.mask else None
        for layer in self.layers: inp = layer(inp, mask=mask)
        return inp
    
class DecoderLayer(nn.Module):
    """Basic block of a Transformer model."""
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, act, resid_p:float=0.,
                 attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, double_drop:bool=True,
                 attn_cls = MultiHeadAttention):
        super().__init__()
        self.mhra = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, 
                             attn_p=attn_p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, act=act, 
                                 ff_p=ff_p, double_drop=double_drop)

        
    def forward(self, x, mask=None, **kwargs): 
        return self.ff(self.mhra(x, mask=mask, **kwargs))
    
def feed_forward(d_model:int, d_ff:int, act, ff_p:float=0., double_drop:bool=False):
    layers = [nn.Linear(d_model, d_ff), act]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return fastai.layers.SequentialEx(*layers, nn.Linear(d_ff, d_model), 
                                      nn.Dropout(ff_p), fastai.layers.MergeLayer())


def convl(ni, nf, ks = 3, stride = 1, padding = None, bias = False,
          bn = False, act_fn = RLReLU(), ortho = True):
    if padding is None: padding = (ks-1)//2
    cv = nn.Conv2d(ni, nf, kernel_size = ks, stride = stride, padding = padding)
    if ortho: cv.weight.data = ortho_weights(cv.weight.data.size(), scale = np.sqrt(2))
    layers = [cv]
    
    if act_fn is not None:
        layers += [act_fn]
        
    if bn: 
        bn = nn.BatchNorm2d(nf)
        layers += [bn]

    return nn.Sequential(*layers)

def atari_initializer(module):
    """ Parameter initializer for Atari models
    Initializes Linear, Conv2d.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

def ortho_weights(shape, scale=1.):
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError
        
    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))
