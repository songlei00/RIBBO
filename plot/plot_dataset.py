from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt 

from datasets.datasets import TrajectoryDataset


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

def plot_dataset(
    dataset: TrajectoryDataset,
    xy_fn,
    split_fn,
    group_fn,
    nrows=None,
    ncols=None,
    average_group=True,
    shaded_std=True,
    xlabel=None,
    ylabel=None,
):
    trajectories = dataset.trajectory_list

    sk2t = defaultdict(list) # splitkey2trajectory
    for trajectory in trajectories:
        k = split_fn(trajectory)
        sk2t[k].append(trajectory)

    if nrows is None and ncols is None:
        nrows, ncols = 1, len(sk2t)
    elif nrows is None:
        nrows = np.ceil(len(sk2t) / ncols).astype(int)
    else:
        ncols = np.ceil(len(sk2t) / nrows).astype(int)
    figsize = (8 * ncols, 6 * nrows)
    f, axs = plt.subplots(nrows, ncols, figsize=figsize)

    groups = list(set(group_fn(trajectory) for trajectory in trajectories))
    groups.sort()

    for i, sk in enumerate(sorted(sk2t.keys())):
        idx_row = i // ncols
        idx_col = i % ncols
        ax = axs[idx_row][idx_col]

        g2l = {} # group2line
        g2c = defaultdict(int) # group2count
        g2t = defaultdict(list) # group2trajectory
        sub_trajectories = sk2t[sk]

        for trajectory in sub_trajectories:
            group_key = group_fn(trajectory)
            color = COLORS[groups.index(group_key) % len(COLORS)]
            g2c[group_key] += 1
            x, y = xy_fn(trajectory)

            if average_group:
                g2t[group_key].append((x, y))
            else:
                l, = ax.plot(x, y, color=color)
                g2l[group_key] = l

        if average_group:
            for group_key in groups:
                color = COLORS[groups.index(group_key) % len(COLORS)]
                xys = g2t[group_key]
                if not any(xys):
                    continue
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                # assert set(origxs) == 1
                usex = origxs[0]
                ys = [xy[1][: minxlen] for xy in xys]

                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                l, = ax.plot(usex, ymean, color=color)
                g2l[group_key] = l

                if shaded_std:
                    ax.fill_between(usex, ymean-ystd, ymean+ystd, color=color, alpha=0.2)

        plt.tight_layout()
        ax.legend(g2l.values(), g2l.keys(), loc='lower right')
        ax.set_title(sk)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        print('--- splitkey: {} ---'.format(sk))
        print(g2c)

    return f, axs
