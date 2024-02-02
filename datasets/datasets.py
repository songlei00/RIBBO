import os
import pickle
from typing import List, Optional, Union
from collections import defaultdict

from tqdm import trange
import torch
import numpy as np
from typing import List, Optional
from torch.utils.data import Dataset, IterableDataset

from datasets.trajectory import Trajectory
from datasets.metrics import metric_regret
from datasets.load_datasets import load_trajectory_dataset
from algorithms.data_filter import filter_designer
from data_augment.operators import operator_list


class TrajectoryDataset():
    def __init__(
        self, 
        search_space_id: Union[str, List[str]],
        data_dir: str,
        cache_dir: str, 
        input_seq_len: int=300, 
        max_input_seq_len: int=300,
        normalize_method: str="random",  # choices are ["random", "dataset", "none"]
        scale_clip_range: Optional[List[float]]=None, 
        augment: bool = False,
        update: bool = False,
        filter_data: bool = False,
        n_block: int = 1,
    ) -> None:
        if isinstance(search_space_id, str):
            search_space_id = [search_space_id]
        cache_dirs = []
        for id in search_space_id:
            cache_dirs.append(os.path.join(cache_dir, id))

        block_size = 5
        trajectory_list = []
        for id, cache_dir in zip(search_space_id, cache_dirs):
            if not os.path.exists(cache_dir) or update:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                t = self.create_cache(id, data_dir, cache_dir, block_size)
            else:
                t = self.load_cache(id, cache_dir, block_size, n_block)
                assert isinstance(t, list)
            trajectory_list.extend(t)
        self.trajectory_list = trajectory_list
        if filter_data:
            self.trajectory_list = filter_designer(self.trajectory_list)
        
        designer_cnt = dict()
        for t in self.trajectory_list:
            designer = t.metadata['designer']
            designer_cnt[designer] = designer_cnt.get(designer, 0) + 1

        print('===== trajectory info =====')
        print('total:', len(self.trajectory_list))
        for k, v in designer_cnt.items():
            print(k, v)
        print('===========================')

        for t in self.trajectory_list:
            t.X = t.X[: max_input_seq_len]
            t.y = t.y[: max_input_seq_len]
            t.metadata['length'] = max_input_seq_len

        print('Trajectory len:', len(self.trajectory_list[0].X))

        if augment:
            augment_trajectory_list = []
            augment_ratio = 1
            augment_num = int(augment_ratio * len(self.trajectory_list))
            idx = np.random.choice(len(self.trajectory_list), augment_num, replace=True)
            for i in trange(idx):
                t = self.trajectory_list[i]
                op_i = np.random.choice(len(operator_list), 1, replace=False)
                op = operator_list[op_i]
                augment_t = op(t)
                augment_trajectory_list.append(augment_t)

            self.trajectory_list += augment_trajectory_list
            print('After augmentation, len: {}'.format(len(self.trajectory_list)))

        # group by search_space_id and dataset_id
        # trajectory_dict = dict()
        # for t in self.trajectory_list:
        #     key = t.metadata['search_space_id'] + t.metadata['dataset_id']
        #     if key not in trajectory_dict:
        #         trajectory_dict[key] = []
        #     trajectory_dict[key].append(t)
        # self.trajectory_dict = trajectory_dict
        
        # get raw metrics
        self.id2info, self.sp_id2info, self.global_info = self.get_dataset_info()
        self.algo = {
            "Random": 0, 
            "ShuffledGridSearch": 1, 
            "RegularizedEvolution": 2, 
            "HillClimbing": 3, 
            "EagleStrategy": 4, 
            "CMAES": 5,
            "BotorchBO": 6,
            # "HeBO": 7, 
            # "PyCMA": 8,
        }
        
        # calculate regrets
        self.set_regrets()
        self.input_seq_len = input_seq_len
        self.normalize_method = normalize_method
        self.scale_clip_range = scale_clip_range

    def create_cache(self, search_space_id, data_dir, cache_dir, block_size=50):
        trajectory_list = load_trajectory_dataset(data_dir, search_space_id)

        k2t = dict()
        for t in trajectory_list:
            seed = t.metadata['seed']
            idx = int(seed) // block_size
            if idx in k2t:
                k2t[idx].append(t)
            else:
                k2t[idx] = [t]

        for k in k2t:
            cache_path = os.path.join(cache_dir, '{}_{}_{}.pkl'.format(search_space_id, k*block_size, (k+1)*block_size-1))
            with open(cache_path, 'wb') as f:
                pickle.dump(k2t[k], f)
            print('Save trajectory to {}'.format(cache_path))

        return trajectory_list
        
    def load_cache(self, search_space, cache_dir, block_size=50, n_block=None):
        trajectory_list = []

        if n_block is None:
            cache_files = [file for file in os.listdir(cache_dir) if file.startswith(search_space)]
        else:
            cache_files = ['{}_{}_{}.pkl'.format(search_space, i*block_size, (i+1)*block_size-1) for i in range(n_block)]

        for file in cache_files:
            cache_path = os.path.join(cache_dir, file)

            if not os.path.exists(cache_path):
                print('{} not exists'.format(cache_path))
                continue

            with open(cache_path, 'rb') as f:
                t_list = pickle.load(f)
                print('Load trajectory from {}, size: {}'.format(cache_path, len(t_list)))

            trajectory_list.extend(t_list)
        return trajectory_list

    def get_dataset_info(self):
        def get_trajectory_list_x_max(t_list):
            return max(t.X.max() for t in t_list).item()
        def get_trajectory_list_x_min(t_list):
            return min(t.X.min() for t in t_list).item()
        def get_trajectory_list_y_max(t_list):
            return max(t.y.max() for t in t_list).item()
        def get_trajectory_list_y_min(t_list):
            return min(t.y.min() for t in t_list).item()
        
        id2info = dict() # search_spce_id -> dataset_id -> info
        sp_id2info = dict() # search_space_id -> info
        global_info = dict()

        # group the trajectory by id, and calc the metrics
        grouped_trajectory_dict = dict()
        for t in self.trajectory_list:
            sp_id = t.metadata['search_space_id']
            ds_id = t.metadata['dataset_id']
            if sp_id not in grouped_trajectory_dict:
                grouped_trajectory_dict[sp_id] = dict()
            if ds_id not in grouped_trajectory_dict[sp_id]:
                grouped_trajectory_dict[sp_id][ds_id] = []
            grouped_trajectory_dict[sp_id][ds_id].append(t)

        # id info
        for sp_id in grouped_trajectory_dict:
            for ds_id in grouped_trajectory_dict[sp_id]:
                t_list = grouped_trajectory_dict[sp_id][ds_id]

                x_max = get_trajectory_list_x_max(t_list)
                x_min = get_trajectory_list_x_min(t_list)
                y_max = get_trajectory_list_y_max(t_list)
                y_min = get_trajectory_list_y_min(t_list)

                # best_y_average = sum(t.y.max() for t in id2group[id]) / len(id2group[id])
                # span = [t.y.max() - t.y.min() for t in id2group[id]]
                # span_min = min(span).item()
                # span_max = max(span).item()
                # span_mean = np.mean(span).item()

                if sp_id not in id2info:
                    id2info[sp_id] = dict()
                id2info[sp_id][ds_id] = {
                    'x_max': x_max,
                    'x_min': x_min,
                    'y_max': y_max,
                    'y_min': y_min,
                }

        # search space info
        for sp_id in grouped_trajectory_dict:
            sp_t_list = []
            for ds_id in grouped_trajectory_dict[sp_id]:
                sp_t_list.extend(grouped_trajectory_dict[sp_id][ds_id])
            x_max = get_trajectory_list_x_max(sp_t_list)
            x_min = get_trajectory_list_x_min(sp_t_list)
            y_max = get_trajectory_list_y_max(sp_t_list)
            y_min = get_trajectory_list_y_min(sp_t_list)
            y_max_mean = sum([id2info[sp_id][ds_id]['y_max'] for ds_id in id2info[sp_id]]) / len(id2info[sp_id])
            y_min_mean = sum([id2info[sp_id][ds_id]['y_min'] for ds_id in id2info[sp_id]]) / len(id2info[sp_id])

            sp_id2info[sp_id] = {
                'x_max': x_max,
                'x_min': x_min,
                'y_max': y_max,
                'y_min': y_min,
                'y_max_mean': y_max_mean,
                'y_min_mean': y_min_mean,
            }
        
        # global info
        x_max = max([sp_id2info[sp_id]['x_max'] for sp_id in sp_id2info])
        x_min = min([sp_id2info[sp_id]['x_min'] for sp_id in sp_id2info])
        y_max = max([sp_id2info[sp_id]["y_max"] for sp_id in sp_id2info])
        y_min = min([sp_id2info[sp_id]["y_min"] for sp_id in sp_id2info])
        y_max_sum, y_min_sum, cnt = 0, 0, 0
        for sp_id in id2info:
            for ds_id in id2info[sp_id]:
                y_max_sum += id2info[sp_id][ds_id]['y_max']
                y_min_sum += id2info[sp_id][ds_id]['y_min']
                cnt += 1
        y_max_mean = y_max_sum / cnt
        y_min_mean = y_min_sum / cnt
        global_info = {
            "x_min": x_min, 
            "x_max": x_max, 
            "y_min": y_min, 
            "y_max": y_max, 
            "y_max_mean": y_max_mean, 
            "y_min_mean": y_min_mean, 
            'search_space_ids': sorted(list(sp_id2info.keys())),
        }
        return id2info, sp_id2info, global_info

    def set_regrets(self):
        for i in range(len(self.trajectory_list)):
            sp_id = self.trajectory_list[i].metadata['search_space_id']
            ds_id = self.trajectory_list[i].metadata['dataset_id']
            y_max = self.id2info[sp_id][ds_id]['y_max']
            self.trajectory_list[i].regrets = metric_regret(self.trajectory_list[i], y_max)
        
    def transform_x(self, fn):
        for i in range(len(self.trajectory_list)):
            self.trajectory_list[i].X = fn(self.trajectory_list[i].X)
            
    def __len__(self):
        return len(self.trajectory_list)
    
    def __getitem__(self, idx):
        trajectory = self.trajectory_list[idx]
        traj_len = trajectory.X.shape[0]
        start_idx = np.random.choice(traj_len+1-self.input_seq_len)
        
        timesteps = torch.arange(start_idx, start_idx+self.input_seq_len)
        
        y, regrets = self.normalize_y_and_regrets(trajectory)
        return {
            "x": trajectory.X[start_idx:start_idx+self.input_seq_len], 
            "y": y[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            "regrets": regrets[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            # "algo": torch.LongTensor([self.algo[trajectory.metadata["designer"]]]), 
            "algo": torch.LongTensor([self.algo.get(trajectory.metadata['designer'], -1)]), 
            "timesteps": timesteps, 
            "masks": torch.ones_like(timesteps).float()
        }
        
    def normalize_y_and_regrets(self, t):
        if self.normalize_method == "none":
            return t.y, t.regrets
        elif self.normalize_method == "random":
            sp_id, ds_id = t.metadata['search_space_id'], t.metadata['dataset_id']
            dataset_y_min, dataset_y_max = self.id2info[sp_id][ds_id]['y_min'], self.id2info[sp_id][ds_id]['y_max']
            span = (dataset_y_max - dataset_y_min + 1e-6) / 2.0
            l = np.random.uniform(low=dataset_y_min-span/2, high=dataset_y_min+span/2)
            h = np.random.uniform(low=dataset_y_max-span/2, high=dataset_y_max+span/2)
            scale = h-l
            if self.scale_clip_range is not None:
                scale = np.clip(scale, self.scale_clip_range[0], self.scale_clip_range[1])
            return (t.y-l) / scale, t.regrets / scale
        elif self.normalize_method == "dataset": 
            sp_id, ds_id = t.metadata['search_space_id'], t.metadata['dataset_id']
            dataset_y_min, dataset_y_max = self.id2info[sp_id][ds_id]['y_min'], self.id2info[sp_id][ds_id]['y_max']
            scale = dataset_y_max - dataset_y_min + 1e-6
            if self.scale_clip_range is not None:
                scale = np.clip(scale, self.scale_clip_range[0], self.scale_clip_range[1])
            return (t.y - dataset_y_min) / scale, t.regrets / scale
            

class TrajectoryDictDataset(TrajectoryDataset, Dataset):
    def __init__(
        self, 
        search_space_id: Union[str, List[str]], 
        data_dir: str, 
        cache_dir: str, 
        input_seq_len: int=300, 
        normalize_method: str="random", 
        scale_clip_range: Optional[List[float]]=None, 
        augment: bool=False,
        update: bool=False, 
        filter_data: bool = False,
        n_block: int = 1,
    ):
        TrajectoryDataset.__init__(
            self,
            search_space_id,
            data_dir,
            cache_dir,
            input_seq_len,
            normalize_method,
            scale_clip_range,
            augment=augment,
            update=update,
            filter_data=filter_data,
            n_block=n_block,
        )
        Dataset.__init__(self)
        
    def __getitem__(self, idx):
        return super().__getitem__(self, idx)
        
class TrajectoryIterableDataset(TrajectoryDataset, IterableDataset):
    def __init__(
        self, 
        search_space_id: Union[str, List[str]], 
        data_dir: str, 
        cache_dir: str, 
        input_seq_len: int=300, 
        max_input_seq_len: int=300,
        normalize_method: str="random", 
        scale_clip_range: Optional[List[float]]=None, 
        augment: bool=False,
        update: bool=False, 
        prioritize: bool=False, 
        prioritize_alpha: float=1.0, 
        filter_data: bool = False,
        n_block: int = 1,
    ):
        TrajectoryDataset.__init__(
            self,
            search_space_id,
            data_dir,
            cache_dir,
            input_seq_len,
            max_input_seq_len,
            normalize_method,
            scale_clip_range,
            augment=augment,
            update=update,
            filter_data=filter_data,
            n_block=n_block,
        )
        IterableDataset.__init__(self)
        
        self.prioritize = prioritize
        if prioritize:
            from UtilsRL.data_structure import SumTree, MinTree
            self.prioritize_alpha = prioritize_alpha
            self.sum_tree = SumTree(len(self))
            self.sum_tree.reset()
            self.metric_fn = lambda x: np.abs(x) ** prioritize_alpha
            
            metric_values = []
            for i, t in enumerate(self.trajectory_list):
                sp_id, ds_id = t.metadata['search_space_id'], t.metadata['dataset_id']
                span = self.id2info[sp_id][ds_id]["y_max"] - self.id2info[sp_id][ds_id]["y_min"]
                # span_ratio = self.id2info[dataset_id]["span_max"] / self.id2info[dataset_id]["span_mean"]
                # metric_values.append(self.metric_fn(span_ratio))
                metric_values.append(self.metric_fn(span))
            metric_values = np.asarray(metric_values, dtype=np.float32)
            self.sum_tree.add(metric_values)
            
    def __iter__(self):
        while True:
            if self.prioritize:
                _target = np.random.random()
                idx = self.sum_tree.find(_target)[0]
            else:
                idx = np.random.choice(len(self))
            yield super().__getitem__(idx)
