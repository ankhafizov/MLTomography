import numpy as np
from scipy.ndimage import gaussian_filter
from porespy.tools import norm_to_uniform


def gen_phantom(shape, porosity, sigma):
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    img = np.random.random(shape)
    img = gaussian_filter(img, sigma=sigma)
    img = norm_to_uniform(img, scale=[0, 1])
    if porosity:
        img = img > porosity
    return img