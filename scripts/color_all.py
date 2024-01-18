import os

import numpy as np
import matplotlib.pyplot as plt

def plot_legend(n_col=5):
    output_path = './plot/rollout/'
    print(f"Saving legend to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    COLORS = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
        'purple', 'pink', 'brown', 'orange', 'teal',  'lightblue', 'lime',
        'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold',
        'darkred', 'darkblue'
    ]

    labels = colors = COLORS
    colors = [c if isinstance(c, str) else tuple([i/255 for i in c]) for c in colors]
    n = len(colors)
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(n)]

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

    fig.savefig(os.path.join(output_path, "color_all.pdf"), bbox_inches=bbox)

if __name__ == '__main__':
    plot_legend(n_col=5)