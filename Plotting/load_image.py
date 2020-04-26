import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

# load already prepared ndarray from scipy
ascent = scipy.misc.ascent()

# see the default colormap to gray
plt.gray()

plt.imshow(ascent)
plt.colorbar()
plt.show()

print(ascent.shape)
print(ascent.max())
print(ascent.dtype)

bug = Image.open('./data/stinkbug.png')

# 3D array
arr = np.array(bug.getdata(), np.uint8).reshape(bug.size[1], bug.size[0], 3)
plt.gray()
plt.imshow(arr)
plt.colorbar()
plt.show()

bug2 = imageio.imread('./data/stinkbug.png')

# inspect the shape of the image
print(bug2.shape)

# the original image is RGB having values for all three
# channels separately. For simplicity, we convert that to greyscale image
# by picking up just one channel
bug2 = bug2[:, :, 0]

plt.figure()
plt.gray()

plt.subplot(121)
plt.imshow(bug2)

# show 'zoomed' region
zbug = bug2[100:350, 140:350]

plt.subplot(122)
plt.imshow(zbug)
plt.show()

filename = './data/stinkbug.png'
image = np.memmap(filename, dtype=np.uint8, shape=(375, 500))
