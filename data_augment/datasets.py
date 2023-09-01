import torch
from torch.utils.data import IterableDataset


class AugmentedDataset(IterableDataset):
    def __init__(self, aug_prob=0.1):
        super(AugmentedDataset).__init__()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pass
        else:
            pass
        return 