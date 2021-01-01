import numpy as np
from scipy.ndimage import gaussian_filter
from porespy.tools import norm_to_uniform


def gen_phantom(shape, porosity, blobiness):
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    im = np.random.random(shape)
    im = gaussian_filter(im, sigma=blobiness)
    im = norm_to_uniform(im, scale=[0, 1])
    if porosity:
        im = im > porosity
    return im