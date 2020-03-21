import matplotlib.pyplot as plt
import porespy
import model_of_experiment as moe
import numpy as np
import data_manager as dm


def get_min_max(img):

    img_min = img.min()
    img_max = img.max()

    print(f'min: {img_min}, max: {img_max}')

    return img_min, img_max


def create_phantom_and_process(shape, porosity, blobiness, noise, num_of_angles, tag, preview=False, noise_method='s&p'):
    """
    Generates and saves phantom to the database.

    Parameters:
    -----------
    shape: array, dtype = int.
        Shape of original phantom, must contain 2 or 3 int numbers.

    processed_phantom: ndarray.
        Phantom object which represents result of the experiment

    porosity: float.
        Phantom's porosity

    blobiness: int.
        Phantom's blobiness

    noise: float.
        Noise probability for SALT AND PEPPER algorithm — noise_method='s&p'
        Intensity value for POISSON noise algorithm — noise_method='poisson'

    num_of_angles: int.
        Number of Radon projections

    tag: 'test', 'train' or another
        This parameter controls conflicts if several csv files are generated for 1 phanom.
        Keep it different for staging different images with similar parameters

    preview: bool
        shows images of phantoms if True

    results:
    --------
    out: (array, array)
        (generated phantom, processed phantom)
    """

    print(f'shape { shape }, porosity { porosity }, blobiness { blobiness }, noise { noise }')

    phantom = porespy.generators.blobs(shape, porosity=porosity, blobiness=blobiness)
    phantom = porespy.filters.trim_floating_solid(phantom)

    # invert phantom to set stone/pore coding as following: 1 — stone, 0 — pore
    phantom = ~phantom

    processed_phantom, phantom = moe.process_image(phantom, num_of_angles, noise, noise_method)

    print("processed_phantom shape: ", processed_phantom.shape)

    pp_min, pp_max = get_min_max(processed_phantom)
    print('norm image from 0 to 1')
    processed_phantom = (processed_phantom - pp_min) / (pp_max - pp_min)
    get_min_max(processed_phantom)

    dm.save(phantom, processed_phantom, porosity, blobiness, noise, num_of_angles, tag)
    if preview:
        if len(phantom.shape) == 3:
            _, axes = plt.subplots(1, 2, figsize=(10, 10))
            axes[0].imshow(processed_phantom[:, :, 0], cmap='gray')
            axes[1].imshow(phantom[:, :, 0], cmap='gray')
        else:
            _, axes = plt.subplots(1, 2, figsize=(10, 10))
            axes[0].imshow(processed_phantom, cmap='gray')
            axes[1].imshow(phantom, cmap='gray')

    return np.absolute(phantom), np.absolute(processed_phantom)  # why return np.absolute?


def main():

    shape = (500, 500)
    porosity = 0.3
    blobiness = 2
    noise = 0.05
    num_of_angles = 180
    tag = 'train'

    orig_phantom_train, proc_phantom_train = create_phantom_and_process(shape, porosity, blobiness, noise, num_of_angles, tag)
    print(dm.show_data_info())

    plt.show()


if __name__ == '__main__':
    main()
