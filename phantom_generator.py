import numpy as np
from scipy.ndimage import gaussian_filter
from porespy.tools import norm_to_uniform
import random


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output


def generate_phantom(shape, porosity, sigma, noise=0):
    """
    noise=0 - бинарный исходный фантом
    
    noise>0 - есть зашумление (рекомендую noise = 0.1)
    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    img = np.random.random(shape)
    img = gaussian_filter(img, sigma=sigma)
    img = norm_to_uniform(img, scale=[0, 1])
    if porosity:
        img = img > porosity
    
    img = sp_noise(img, noise).astype(bool)
    return img
