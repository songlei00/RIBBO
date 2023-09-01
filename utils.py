import random
import os
import logging
from typing import Dict

import torch
import numpy as np

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.enable = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPASE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    log.info(f'Global seed set to {seed}')
    
    return seed


def print_dict(d: Dict, depth: int = 0) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            log.info('|  '*depth + '|- {}'.format(k))
            print_dict(v, depth+1)
        else:
            log.info('|  '*depth + '|- ' + '{}: {}'.format(k, v))