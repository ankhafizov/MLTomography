
import numpy as np


def invert_signal(sig):
    sig = np.asarray(sig, dtype=int)
    return sig.max() - sig