def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=0.05, ypos=0.95, fs=50):

    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        # Axes3d
        try:
            ax.text2D(xpos, ypos, letters[idx].capitalize(),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='demibold',
                    fontsize=fs,
                    transform=ax.transAxes)
        except AttributeError:
            ax.text(xpos, ypos, letters[idx].capitalize(),
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontweight='demibold',
                      fontsize=fs,
                      transform=ax.transAxes)


def simplify_axes(axes, remove_bottom=False):

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        if remove_bottom:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])


def color_axes(axes, clr):
    if not type(axes) is list:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis='x', colors=clr)
        ax.tick_params(axis='y', colors=clr)
        for spine in ax.spines.values():
            spine.set_edgecolor(clr)