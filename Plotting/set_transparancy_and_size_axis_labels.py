import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

data = np.random.randn(70)
plt.plot(data)

fontsize = 18
title = "This is figure title"
x_label = "This is x axis label"
y_label = "This is y axis label"

title_text_obj = plt.title(title, fontsize=fontsize, verticalalignment='bottom')
# add the default patheffects with no parameters
title_text_obj.set_path_effects([patheffects.withSimplePatchShadow()])

# offset -- set the 'angle' of the shadow, position of the shadow to the parent object
# shadow_rgbFace -- set the color of shadow
# alpha -- setup the transparency of the shadow
offset = (1, -1)
rgbRed = (1.0, 0.0, 0.0)
alpha = 0.8

# customize shadow properties
pe = patheffects.withSimplePatchShadow(offset=offset, shadow_rgbFace=rgbRed, alpha=alpha)

# apply them to the xaxis and yaxis labels
xlabel_obj = plt.xlabel(x_label, fontsize=fontsize, alpha=0.5)
xlabel_obj.set_path_effects([pe])

ylabel_obj = plt.ylabel(y_label, fontsize=fontsize, alpha=0.5)
ylabel_obj.set_path_effects([pe])

plt.show()
