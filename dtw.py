import numpy as np
from cdtw import pydtw


def dtw(sig1, sig2, window=5):
    return pydtw.dtw(sig1, sig2, pydtw.Settings(step='p0sym', window='palival', param=window, norm=False, compute_path=False)).get_dist()


def dtw_fast(sig1, sig2, window=20):
    """
    Algoritmus Fast Dynamic Time Warping  pro dva signaly.
    :param array sig1: signal 1
    :param array sig2: signal 2
    :param window: vymezuje rozmezi pruchodu pro osu x v matici

    :return: vzdálenost signálů
    """
    if len(sig2) > len(sig1):   # optimalizace aby y >= x
        x = sig1
        sig1 = sig2
        sig2 = x

    xlen = len(sig2)
    ylen = len(sig1)
    D = np.full((ylen, xlen), np.iinfo(np.int32).max, dtype=np.int32)
    ratio = xlen / ylen   # pomer delek signalu kvuli zkoseni matice

    for i in range(ylen):
        j = int(i*ratio)
        l = max(0, j - window)
        u = j + window + 1
        D[i, l:u] = np.absolute(sig2 - sig1[i])[l:u]

    D[0, :window+1] = np.cumsum(D[0, :window+1])

    yindex = min(ylen-1, int(window/ratio))     # odhad
    while D[yindex, 0] == np.iinfo(np.int32).max:
        yindex -= 1
    while yindex < ylen and D[yindex, 0] != np.iinfo(np.int32).max:
        yindex += 1

    D[:yindex, 0] = np.cumsum(D[:yindex, 0])

    for i in range(1, ylen):
        p = int(i * ratio)
        l = max(p - window, 1)
        u = min(p + window + 1, xlen)

        for j in range(l, u):
            m = min([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            D[i, j] += m

    return D[ylen - 1, xlen - 1]


def dtw_basic(sig1, sig2):
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



