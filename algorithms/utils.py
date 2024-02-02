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


def calculate_metrics(id2y, id2normalized_y, id2onestep_normalized_regret, id2x=None):
    metrics = {}
    
    # best y: max over sequence, average over eval num
    # best_y_sum, best_normalized_y_sum, normalized_regret_sum = 0, 0, 0
    # for id in id2normalized_y:
    #     best_y_this = id2y[id].max(axis=1).mean()
    #     metrics["best_y_"+id] = best_y_this
    #     best_y_sum += best_y_this

    #     best_normalized_y_this = id2normalized_y[id].max(axis=1).mean()
    #     metrics["best_normalized_y_"+id] = best_normalized_y_this
    #     best_normalized_y_sum += best_normalized_y_this
    #     
    #     regret_this = id2onestep_normalized_regret[id].sum(axis=1).mean()
    #     metrics["normalized_regret_"+id] = regret_this
    #     normalized_regret_sum += regret_this

    # metrics["best_y_agg"] = best_y_sum / len(id2y)
    # metrics["best_normalized_y_agg"] = best_normalized_y_sum / len(id2normalized_y)
    # metrics["normalized_regret_agg"] = normalized_regret_sum / len(id2onestep_normalized_regret)

    trajectory_record = {}

    # mean y, mean normalized y
    for id in id2normalized_y:
        trajectory_record[id] = {}
        if id2x is not None:
            trajectory_record[id]['X'] = id2x[id]
        trajectory_record[id]['y'] = id2y[id].squeeze(-1)
        trajectory_record[id]['normalized_y'] = id2normalized_y[id].squeeze(-1)
        trajectory_record[id]['normalized_regret'] = id2onestep_normalized_regret[id].squeeze(-1)

    return metrics, trajectory_record
