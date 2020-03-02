import matplotlib.pyplot as plt
import porespy.generators as generator
import model_of_experiment as moe
import shelve

def create_phantom_and_process(preview = True):
    phantom = generator.blobs(shape, porosity=porsty, blobiness=blobns)
    processed_phantom = moe.process_image(phantom, angles, noise_prob)

    if preview:
        _, axes = plt.subplots(1,2)
        axes[0].imshow(processed_phantom[:,:,0], cmap = 'gray')
        axes[1].imshow(phantom[:,:,0], cmap = 'gray')
    
    return phantom, processed_phantom


def save(orig_phantom, processed_phantom, tag):
    key = f"dim{len(shape)},porsty{porsty},blobns{blobns},noise{noise_prob},angles{angles}_{tag}"
    files = {'original': orig_phantom,
            'processed': processed_phantom}
    db = shelve.open('database')
    db[key] = files
    db.close()


shape = (50, 50, 50)
porsty = 0.3
blobns = 2
noise_prob = 0.01
angles = 360

tag = 'train'
# tag = 'test'
save(*create_phantom_and_process(), tag)

plt.show()