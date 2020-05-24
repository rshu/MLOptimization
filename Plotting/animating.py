import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


def init():
    """Clear current frame"""
    line.set_data([], [])
    return line,


def animate(i):
    """Draw figure"""
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i)) * np.cos(22 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,


# This call puts the work in motion
# Connecting init and animate functions and figure we want to draw
# Set blit to False if in OS X
animator = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

# This call creates the video file.
# Temporarily, create frame is saved as PNG file
# and later processed by ffmpeg encoder into MPEG4 file
# we can pass vairous arguments to ffmpeg via extra_args
# animator.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'], writer='ffmpeg_file')

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Rui'), bitrate=1800)
animator.save("basic_animation.mp4", writer=writer)

plt.show()
