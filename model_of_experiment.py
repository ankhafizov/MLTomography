import numpy as np
import random
from skimage.transform import radon, iradon


def process_image(image, angles, noise, noise_method):
    print("phantom shape: ", image.shape)
    sim = create_sinogram(angles, image)
    print("sinogram shape: ", sim.shape)
    if noise_method == 's&p':
        sim = add_noise(sim, noise)
    elif noise_method == 'poisson':
        sim = add_poisson_noise(sim, noise)
    rec = reconstruct(sim)
    print("reconstruction shape: ", rec.shape)
    return crop_image(rec, image.shape), crop_image(image, image.shape)


def left_edge(center, halves, index):
    return np.ceil(center[index] - halves[index]).astype(np.int)


def right_edge(center, halves, odds, index):
    return np.ceil(center[index] + halves[index] + odds[index]).astype(np.int)


def crop_image(rec_image, new_shape):

    center = [x // 2 for x in rec_image.shape]
    halves = [x // 2 for x in new_shape]
    odds = [x % 2 for x in new_shape]

    ranges = (
        slice(
            left_edge(center, halves, i),
            right_edge(center, halves, odds, i)
        )
        for i in range(3)
    )

    return rec_image[tuple(ranges)]


def crop(img, new_shape):
    print(f'crop image to shape {new_shape}')
    if len(new_shape)==2 and len(img.shape)==2:
        cen_x, cen_y = np.asarray(img.shape)//2
        x_len, y_len = np.asarray(new_shape)//2-1
        cropped_img = img[(cen_x-x_len):(cen_x+x_len), (cen_y-y_len):(cen_y+y_len)]
    elif len(new_shape)==3 and len(img.shape)==3:
        cen_x, cen_y, cen_z = np.asarray(img.shape)//2
        x_len, y_len, z_len = np.asarray(new_shape)//2-1
        cropped_img = img[(cen_x-x_len):(cen_x+x_len), 
                        (cen_y-y_len):(cen_y+y_len), 
                        (cen_z-z_len):(cen_z+z_len)]
    else:
        print("Mistake: shape of img does not fit new shape (2 or 3)")
        cropped_img = None

    print(f'cropped image have shape {cropped_img.shape}')
    return cropped_img


def create_sinogram(angles, img):
    set_of_angles = np.linspace(0, 180, angles, endpoint=False)
    if len(img.shape) == 3:
        sim = np.asarray([radon(img_slice, theta=set_of_angles, circle=False) for img_slice in img])
    else:
        sim = radon(img, theta=set_of_angles, circle=False)
    
    return sim


def reconstruct(sinogram, filt_name='shepp-logan'):
    if len(sinogram.shape) == 3:
        set_of_angles = np.linspace(0, 180, sinogram[0].shape[1], endpoint=False)
        image = [iradon(s, set_of_angles, filter=filt_name) for s in sinogram]
    else:
        set_of_angles = np.linspace(0, 180, sinogram.shape[1], endpoint=False)
        image = iradon(sinogram, set_of_angles, filter=filt_name)

    image = np.asarray(image)
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
        if len(image.shape) == 3:
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
        else:
            for k in range(image.shape[1]):
                for i in range(image.shape[0]):
                        rdn = random.random()
                        if rdn < prob:
                            output[i][k] = rdn*coef
                        elif rdn > thres:
                            output[i][k] = (1-rdn)*coef
                        else:
                            output[i][k] = image[i][k]

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
