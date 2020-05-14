import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def setup(layout=None):
    assert layout is not None

    fig = plt.figure()
    ax = fig.add_subplot(layout)
    return fig, ax


def get_signal():
    # generate data, a sine wave
    t = np.arange(0., 2.5, 0.01)
    s = np.sin(5 * np.pi * t)
    return t, s


def plot_signal(t, s):
    line, = axes.plot(t, s, linewidth=3, color='magenta')
    return line


def make_shadow(fig, axes, line, t, s):
    # how many points to move the shadow
    # create an offset object underneath and just a few points away
    # from the original object
    # the point is 1/72 inches
    # we move the offset object 2pt right and 2pt down
    delta = 2 / 72.
    # xtr, ytr: transformation offset
    # scaletr
    offset = transforms.ScaledTranslation(delta, -delta, fig.dpi_scale_trans)
    offset_transform = axes.transData + offset

    # we plot the same data, but now using offset transform
    # zorder -- to render it below the line
    axes.plot(t, s, linewidth=3, color='gray', transform=offset_transform, zorder=0.5 * line.get_zorder())


if __name__ == "__main__":
    fig, axes = setup(111)
    t, s = get_signal()
    line = plot_signal(t, s)
    make_shadow(fig, axes, line, t, s)
    axes.set_title('Shadow effect using an offset transform')
    plt.show()
