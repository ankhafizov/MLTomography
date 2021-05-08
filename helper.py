
import numpy as np


def invert_signal(sig):
    sig = np.asarray(sig, dtype=float)
    return sig.max() - sig