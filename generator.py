import matplotlib.pyplot as plt
import porespy.generators as generator
import model_of_experiment as moe
import numpy as np
import shelve

def create_phantom_and_process(preview = True):
    phantom = generator.blobs(shape, porosity=porsty, blobiness=blobns)
    processed_phantom, phantom = moe.process_image(phantom, angles, noise_prob)

    if preview:
        if len(phantom.shape) == 3:
            _, axes = plt.subplots(1,2)
            axes[0].imshow(processed_phantom[:,:,0], cmap = 'gray')
            axes[1].imshow(phantom[:,:,0], cmap = 'gray')
        else:
            _, axes = plt.subplots(1,2)
            axes[0].imshow(processed_phantom, cmap = 'gray')
            axes[1].imshow(phantom, cmap = 'gray')

    return np.absolute(phantom), np.absolute(processed_phantom)


def save(orig_phantom, processed_phantom, tag):
    key = f"dim{len(shape)},porsty{porsty},blobns{blobns},noise{noise_prob},angles{angles}_{tag}"
    files = {'original': orig_phantom,
            'processed': processed_phantom}
    db = shelve.open('database')
    db[key] = files
    db.close()


shape = (500, 500)
porsty = 0.3
blobns = 2
noise_prob = 0.05
angles = 180

tag = 'train'
save(*create_phantom_and_process(), tag)

tag = 'test'
save(*create_phantom_and_process(), tag)

plt.show()