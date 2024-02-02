#%%
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


params = {
    'lines.linewidth': 1.5,
    'legend.fontsize': 20,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
matplotlib.rcParams.update(params)

def plot_legend(output_path, palette, n_col=5):
    print(f"Saving legend to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    labels, colors = list(palette.keys()), list(palette.values())
    colors = [c if isinstance(c, str) else tuple([i/255 for i in c]) for c in colors]
    n = len(colors)
    # f = lambda m, c, ls: plt.plot([], [], marker=m, color=c, ls=ls)[0]
    # def get_handles(palette):
    #     handles = []
    #     for l, c in zip(labels, colors):
    #         if l.startswith('OptFormer'):
    #             handles.append(f('none', c, '--'))
    #         else:
    #             handles.append(f('none', c, '-'))
    #     return handles
    # handles = get_handles(palette)
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(n)]

    def reorder(h, l, nc):
        new_h, new_l = [], []
        for row_i in range(nc):
            for col_i in range(int(np.ceil(len(h) / nc))):
                idx = col_i * nc + row_i
                if idx < len(h):
                    new_h.append(h[idx])
                    new_l.append(l[idx])
        return new_h, new_l

    handles, labels = reorder(handles, labels, n_col)
    legend = plt.legend(
        handles, labels, loc=3, framealpha=1, frameon=False, 
        ncol=n_col, bbox_to_anchor=(1,1), columnspacing=1
    )
    fig = legend.figure
    fig.canvas.draw()
    expand=[-1, -1, 1, 1]
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(os.path.join(output_path, "legend.pdf"), bbox_inches=bbox)

# palette = {
#     'DT': 'crimson',
#     'BC': (171, 197, 231),
#     'BC-filter': 'royalblue',
#     # # 'BC': 'orange',
#     'Opt-Eagle': (0, 190, 190),
# 
#     # behavior
#     'Random': 'gray',
#     'ShuffledGridSearch': 'blueviolet',
#     'HillClimbing': (141, 84, 71),
#     'RegularizedEvolution': (124, 136, 6),
#     'EagleStrategy': (255, 153, 51),
#     'CMAES': (71, 71, 71),
#     'BotorchBO': (216, 207, 22),
# }

# regret strategy
# palette = {
#     'DT-relabel': 'blue',
#     'DT-clip': 'blue',
#     'DT-none-0': 'blue',
#     'DT-none': 'blue',
# }

# palette = {
#     'OptFormer (ShuffledGridSearch)': 'blueviolet',
#     'OptFormer (HillClimbing)': (141, 84, 71),
#     'OptFormer (RegularizedEvolution)': (124, 136, 6),
#     'OptFormer (EagleStrategy)': (255, 153, 51),
# 
#     'ShuffledGridSearch': 'blueviolet',
#     'HillClimbing': (141, 84, 71),
#     'RegularizedEvolution': (124, 136, 6),
#     'EagleStrategy': (255, 153, 51),
# }
    
palette = {
    # 'DT': 'crimson',
    # 'BC': (171, 197, 231),
    # 'BC-filter': 'royalblue',
    # 'Opt-Eagle': (0, 190, 190),

    # behavior
    # 'Random': 'gray',
    # 'ShuffledGridSearch': 'blueviolet',
    'HillClimbing': (141, 84, 71),
    'RegularizedEvolution': (124, 136, 6),
    'EagleStrategy': (255, 153, 51),
    'CMAES': (71, 71, 71),
    'BotorchBO': (216, 207, 22),

    # algorithm selection
    # 'DT-regret0': 'crimson',
    # 'DT-regret5': 'darkorange',
    # 'DT-regret10': 'royalblue',
    # 'DT-regret30': 'blueviolet',
    # : 'limegreen',

    # input seq len
    # 'DT-len50': 'crimson',
    # 'DT-len25': 'darkorange',
    # 'DT-len100': 'royalblue',
    # 'DT-len150': 'blueviolet',

    # mix method
    # 'DT-concat': 'crimson',
    # 'DT-add': 'darkorange',
    # 'DT-interleave': 'royalblue',

    # model arch
    # 'DT-mid': 'crimson',
    # 'DT-small': 'darkorange',
    # 'DT-large': 'royalblue',

    # normalize method
    # 'DT-random': 'crimson',
    # 'DT-dataset': 'darkorange',
    # 'DT-none': 'royalblue',

    # regret strategy
    # 'DT-relabel': 'crimson',
    # 'DT-clip': 'darkorange',
    # 'DT-none-0': 'royalblue',
    # 'DT-none-10': 'blueviolet',

    # x_type
    # 'DT-stochastic': 'crimson',
    # 'DT-deterministic': 'darkorange',
}

key_map = {
    # 'DT': 'RIBBO',
    # 'BC': 'BC',
    # 'BC-filter': 'BC Filter',
    # 'Opt-Eagle': 'Optformer',

    # behavior
    # 'Random': 'Random Search',
    # 'ShuffledGridSearch': 'Shuffled Grid Search',
    'HillClimbing': 'Hill Climbing',
    'RegularizedEvolution': 'Regularized Evolution',
    'EagleStrategy': 'Eagle Strategy',
    'CMAES': 'CMA-ES',
    'BotorchBO': 'GP-EI',
}

for k in key_map:
    if k in palette:
        palette[key_map[k]] = palette.pop(k)

output_path=f"./plot/rollout/"
plot_legend(output_path, palette, 5)