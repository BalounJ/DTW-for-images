import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SignalWrapper(object):
    """
    Obalová třída uchovávající informace o signálech a původním obrázku.
    """
    def __init__(self, img_path):
        """
        Uchovává informace o ůvodním obrázku a vytvoří k němu odpovídající signál.

        :param int img_path: Cesta k obr.
        """
        self.img_path = img_path
        self.signal = convert_normalized_image_to_signal(load_img_normalized(img_path))

    def getImgPath(self):
        return self.img_path

    def getSignal(self):
        return self.signal


def get_image_height(img):
    """
    Vrátí výšku obrázku.

    :param img: obrázek
    :return: výška obrázku
    """
    return img.shape[0]


def get_image_width(img):
    """
    Vrátí šířku obrázku.

    :param img: obrázek
    :return: šířka obrázku
    """
    return img.shape[1]


def load_set(root_dir):
    """
    Slouží pro načítání setů.

    :param str root_dir: Cesta ke kořenové složce testovacího, trénovacího nebo validačního setu.
    Očekává se, že v kořenové složce jsou pouze složky s názvy slov např.:
    D:\parzivalDB\set_1\test\a-b
    Ve složkách jsou potom obrázky pro dané slovo např:
    D:\parzivalDB\set_1\test\a-b\d-279b-041_03.png

    :return: seznam obsahující [slovo, [SignalWrapper]]
    """
    rtn = []

    for root, dirs, f in os.walk(root_dir):
        for directory in dirs:
            signals = []
            for r, d, files in os.walk(os.path.join(root, directory)):
                for file in files:
                    img_path = os.path.join(r, file)
                    signals.append(SignalWrapper(img_path))
            rtn.append([directory, signals])
    return rtn


def load_img_normalized(img_path):
    """
    :param str img_path: Cesta k obrázku
    :return: normalizovaný obrázek
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = -(img / img.max()) + 1
    return img


def convert_normalized_image_to_signal(img_norm):
    """
    :param str img_norm: normalizovaný obrázek
    :return: signál vytvořený z obrázku
    """
    sig = np.sum(img_norm, axis=0)
    return sig.astype(np.int32)


def plot_image_signal(img_path, sig):
    """
    vykresli obr. a jeho signal
    :param str img_path: Cesta k obrázku
    :param sig: signal
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    f, axarr = plt.subplots(2, sharex="all")
    axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
    axarr[1].plot(sig)
    plt.show()
