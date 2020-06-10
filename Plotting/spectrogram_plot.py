import os
from math import floor, log
from scikits.audiolab import Sndfile
import numpy as np
from matplotlib import pyplot as plt

# do not support python 3.x
soundfile = Sndfile("./data/test.wav")

# define start/stop seconds and compute start/stop frames
start_sec = 0
stop_sec = 5
start_frame = start_sec * soundfile.samplerate
stop_frame = stop_sec * soundfile.samplerate

# go to the start frame of the sound object
soundfile.seek(start_frame)

# read number of frames from start to stop
delta_frames = stop_frame - start_frame
sample = soundfile.read_frames(delta_frames)

map = 'CMRmap'

fig = plt.figure(figsize=(10, 6), )
ax = fig.add_subplot(111)

# define the number points for Fourier transform
NFFT = 128

# define number of data points to overlap for each block
noverlap = 65

pxx, freq, t, cax = ax.specgram(sample, Fs=soundfile.samplerate,
                                NFFT=NFFT, noverlap=noverlap,
                                cmap=plt.get_cmap(map))
plt.colorbar(cax)
plt.xlabel("Times [sec]")
plt.ylabel("Frequency [Hz]")

plt.show()
