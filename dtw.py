import numpy as np


def dtw(sig1, sig2):
    """
    Algoritmus Dynamic Time Warping (DTW) pro dva signaly.
    :param array sig1: signal 1
    :param array sig2: signal 2
    :return: vzdálenost signálů
    """
    D = np.zeros((len(sig1), len(sig2)), dtype=np.int32)

    for i in range(len(sig1)):
        D[i, :] = np.absolute(sig2 - sig1[i])

    D[0, :] = np.cumsum(D[0, :])
    D[:, 0] = np.cumsum(D[:, 0])

    for i in range(1, len(sig1)):
        for j in range(1, len(sig2)):
            m = min([D[i-1, j-1], D[i-1, j], D[i, j-1]])
            D[i, j] += m

    return D[len(sig1) - 1, len(sig2) - 1]



