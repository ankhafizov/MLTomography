from porespy import generators as ps
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.transform import radon, iradon


def process_image(image, angles, noizeProbability):
    sim = create_sinogram(angles, image)
    sim = add_noise(sim, noizeProbability)
    rec = reconstruct(sim, angles)
    return rec


def create_sinogram(angles, img3d):
    set_of_angles = np.linspace(0, 180, angles, endpoint=False)
    return np.asarray([radon(img, theta=set_of_angles, circle=False) for img in img3d])


def reconstruct(sinogram, angles, filtName='shepp-logan'):
    set_of_angles = np.linspace(0, 180, sinogram[0].shape[1], endpoint=False)

    image = [iradon(s, set_of_angles, filter=filtName) for s in sinogram]
    image = np.asanyarray(image)
    return image


def add_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob = prob/2
    coef = max(np.concatenate(image, axis=None))
    output = image
    thres = 1 - prob
    if prob != 0:
        for k in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j][k] = rdn*coef
                    elif rdn > thres:
                        output[i][j][k] = (1-rdn)*coef
                    else:
                        output[i][j][k] = image[i][j][k]

    return output


def add_poisson_noise(sinogram, intensity):

    sinogram /= 100.0
    I_0 = intensity * np.ones(sinogram.shape)
    I_0 = np.random.poisson(lam=I_0).astype('float32')
    sinogram =  np.exp(-sinogram) * I_0

    I_0_str = intensity * np.ones((10, sinogram.shape[0], sinogram.shape[1]))
    I_0_str = np.random.poisson(lam=I_0_str).astype('float32')
    I_0_str = (np.sum(I_0_str, axis=0) / 10.0).reshape((1, sinogram.shape[0], sinogram.shape[1]))
    I_0_arr = np.repeat(I_0_str, sinogram.shape[2], axis=1).reshape(sinogram.shape)

    sinogram = np.abs(np.log(I_0_arr / sinogram))
    sinogram *= 100.0
    sinogram = np.ceil(sinogram)

    return sinogram
