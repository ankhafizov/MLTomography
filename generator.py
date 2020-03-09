import matplotlib.pyplot as plt
import porespy.generators as generator
import model_of_experiment as moe
import numpy as np
import data_manager as dm


def create_phantom_and_process(shape, porsty, blobns, noise, num_of_angles, tag, preview=True, noise_method='s&p'):
    '''
    Generates and saves phantom to the database.
        
    Parameters:
    -----------
    shape: array, dtype = int.
        Shape of riginal phantom, must contain 2 or 3 int numbers.

    processed_phantom: ndarray.
        Phantom object which represents result of the experiment

    porsty: float.
        Phantom's porosiity
    
    blobns: int.
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
    '''

    phantom = generator.blobs(shape, porosity=porsty, blobiness=blobns)
    processed_phantom, phantom = moe.process_image(phantom, num_of_angles, noise, noise_method)

    print("processed_phantom shape: ", processed_phantom.shape)

    dm.save(phantom, processed_phantom, porsty, blobns, noise, num_of_angles, tag)
    if preview:
        if len(phantom.shape) == 3:
            _, axes = plt.subplots(1, 2)
            axes[0].imshow(processed_phantom[:, :, 0], cmap='gray')
            axes[1].imshow(phantom[:, :, 0], cmap='gray')
        else:
            _, axes = plt.subplots(1, 2)
            axes[0].imshow(processed_phantom, cmap='gray')
            axes[1].imshow(phantom, cmap='gray')

    return np.absolute(phantom), np.absolute(processed_phantom)  # why return np.absolute?


if __name__ == '__main__':
    shape = (500, 500)
    porsty = 0.3
    blobns = 2
    noise = 0.05
    num_of_angles = 180
    tag = 'train'

    orig_phantom_train, proc_phantom_train = create_phantom_and_process(shape, porsty, blobns, noise, num_of_angles, tag)
    print(dm.show_data_info())

    plt.show()