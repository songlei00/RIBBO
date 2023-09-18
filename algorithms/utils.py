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


def calculate_metrics(id2y, id2normalized_y, best_original_y, best_y):
    metrics = {}
    
    # best y: max over sequence, average over eval num
    best_y_sum, best_normalized_y_sum = 0, 0
    for id in id2y:
        best_y_this = id2y[id].max(axis=1).mean()
        metrics["best_y_"+id] = best_y_this
        best_y_sum += best_y_this

        best_normalized_y_this = id2normalized_y[id].max(axis=1).mean()
        metrics['best_normalized_y_'+id] = best_normalized_y_this
        best_normalized_y_sum += best_normalized_y_this

    metrics["best_y_agg"] = best_y_sum / len(id2y)
    metrics['best_normalized_y_agg'] = best_normalized_y_sum / len(id2normalized_y)

    # regret: (best_y - y), sum over sequence, average over eval num
    regret_sum, regret_normalized_sum = 0, 0
    for id in id2y:
        regret_this = (best_original_y - id2y[id]).sum(axis=1).mean()
        regret_sum += regret_this
        metrics["regret_"+id] = regret_this

        regret_normalized_this = (best_y - id2normalized_y[id]).sum(axis=1).mean()
        metrics['regret_normalized_'+id] = best_normalized_y_this
        regret_normalized_sum = regret_normalized_this
    metrics["regret_agg"] = regret_sum / len(id2y)
    metrics['regret_normalized_agg'+id] = regret_normalized_sum / len(id2y)

    trajectory_record = {}

    # mean y, mean normalized y
    for id in id2y:
        mean_y = id2y[id].mean(axis=0)
        mean_normalized_y = id2normalized_y[id].mean(axis=0)
        trajectory_record['mean_y_' + id] = mean_y
        trajectory_record['mean_normalized_y_'+id] = mean_normalized_y

    return metrics, trajectory_record