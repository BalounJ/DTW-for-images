import os
from operator import itemgetter

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

        :param String img_path: Cesta k obr.
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


def find_images(query_img_path, set_path, dtw_func, *argf):
    """
    Slouží pro nalezení nejbližších obrázků.

    :param str query_img_path: cesta pro dotazovaný obrázek

    :param str set_path: Cesta ke kořenové složce setu pro vyhledávání.
    Očekává se, že v kořenové složce jsou pouze složky s názvy slov např.:
    D:\parzivalDB\set_1\test\a-b
    Ve složkách jsou potom obrázky pro dané slovo např:
    D:\parzivalDB\set_1\test\a-b\d-279b-041_03.png

    :param dtw_func: funkce dtw pro pouziti
    :param argf: pripadne dalsi parametry pro dtw_func

    :return: seznam dvojic [cesta k obrazku, dist] serazene od nejmensiho podle dist
    """
    qs = SignalWrapper(query_img_path).getSignal()
    data = load_set(set_path)
    rtn = []

    for d in data:
        for sw in d[1]:
            dist = dtw_func(qs, sw.getSignal(), *argf)
            if isinstance(dist, tuple):
                dist = dist[0]
            rtn.append([sw.getImgPath(), dist])
    return sorted(rtn, key=itemgetter(1))  # seradim podle vzdalenosti


def find_images_plot_n(n, query_img_path, set_path, dtw_func, *argf):
    """
    Vykresli n nejpodobnejsich obr.

    :param int n: pocet obr. k zobrazeni

    :param str query_img_path: cesta pro dotazovaný obrázek

    :param str set_path: Cesta ke kořenové složce setu pro vyhledávání.
    Očekává se, že v kořenové složce jsou pouze složky s názvy slov např.:
    D:\parzivalDB\set_1\test\a-b
    Ve složkách jsou potom obrázky pro dané slovo např:
    D:\parzivalDB\set_1\test\a-b\d-279b-041_03.png

    :param dtw_func: funkce dtw pro pouziti
    :param argf: pripadne dalsi parametry pro dtw_func
    """
    data = find_images(query_img_path, set_path, dtw_func, *argf)

    n = min(n, len(data))
    numpy_vertical = cv2.imread(data[0][0], cv2.IMREAD_GRAYSCALE)
    for i in range(1, n):
        img_path = data[i][0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        numpy_vertical = np.hstack((numpy_vertical, np.full((img.shape[0], 5), fill_value=150, dtype=np.uint8), img))

    qimg = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    f, axarr = plt.subplots(2, 1, sharex="all", sharey="all")
    axarr[0].imshow(qimg, cmap=plt.get_cmap('gray'))
    axarr[0].title.set_text("Query")
    axarr[1].imshow(numpy_vertical, cmap=plt.get_cmap('gray'))
    axarr[1].title.set_text("Result")
    plt.show()



