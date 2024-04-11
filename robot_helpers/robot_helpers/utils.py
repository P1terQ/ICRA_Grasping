import numpy as np


def sph2cart(r, theta, phi):
    return np.r_[
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ]
