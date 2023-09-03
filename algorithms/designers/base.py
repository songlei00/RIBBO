import torch.nn as nn

class BaseDesigner(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def configure_optimizers(self, *args, **kwargs):
        raise NotImplementedError
    
    def suggest(self, *args, **kwargs):
        raise NotImplementedError
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        super().to(device)