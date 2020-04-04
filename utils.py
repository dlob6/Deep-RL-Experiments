from collections import namedtuple
import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

    screen = screen[:, int(screen_height*0.3):int(screen_height*0.9), int(screen_width*0.2):int(screen_width*0.8)]
    #screen = screen[:, int(screen_height*0.1):int(screen_height*0.95),:]
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
