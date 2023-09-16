import torch

def get_vector_statistics(v):
    func_dict = {
        'min': torch.min,
        'max': torch.max,
        'mean': torch.mean,
        'median': torch.median,
        'norm': lambda x: torch.norm(x, p=1) / len(x.flatten()),
    }
    ret = dict()
    for key, func in func_dict.items():
        ret[key] = func(v)
    return ret