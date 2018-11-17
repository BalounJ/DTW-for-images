from itertools import chain
from operator import itemgetter

import numpy as np

from dtw import dtw
from lib import load_set


def __count_average_prec(query_res):
    """
    Spočte Average Precision

    :param query_res: Výsledek dotazu obsahující dvojice [bool, dist]

    :return: Average Precision dotazu
    """
    res = []
    query_res = sorted(query_res, key=itemgetter(1))  # seradim podle vzdalenosti
    ok = 0
    total = 0

    for qr in query_res:
        total += 1
        if qr[0]:
            ok += 1
            res.append(ok/total)

    average_prec = np.mean(res)
    return average_prec


def __get_incorrect(query_sig, data, correct_index, max_dist):
    """
    Hledá nesprávná slova, která jsou bližší než max_dist

    :param query_sig: signal se kterým porovnávat
    :param data: List obsahující [slovo, [SignalWrapper]]
    :param max_dist: Pokud je vzdálenost větší, nebude brán v potaz

    :return: Nesprávná slova, která jsou bližší než max_distance, a jejich vzdálenost [False, dist]
    """
    rtn = []

    for i in chain(range(0, correct_index), range(correct_index + 1, len(data))):
        item = data[i]
        sigs = item[1]
        for sig in sigs:
            dist = dtw(query_sig, sig.getSignal())
            if dist <= max_dist:
                rtn.append([False, dist])

    return rtn


def __get_average_prec(index, data):
    """
    Spočte Average Precision pro jeden dotaz.

    :param index: Index pro dotaz
    :param data: List obsahující [slovo, [SignalWrapper]]

    :return: Average Precision pro dotaz
    """
    max_dist = 0
    correct = []

    item = data[index]
    correct_sigs = item[1]

    query_sig = correct_sigs[0].getSignal()

    for i in range(1, len(correct_sigs)):  # nejprve ohodnotim spravne obrazky, prvni vynecham
        sig = correct_sigs[i].getSignal()
        dist = dtw(query_sig, sig)
        correct.append([True, dist])  # pridam spravny a vzdalenost
        max_dist = max(max_dist, dist)

    query_res = __get_incorrect(query_sig, data, index, max_dist)  # nespravna slova, jejichz vzdalenost je mensi nez nejhorsi spravna
    query_res.extend(correct)                                               # pridam spravna slova
    average_prec = __count_average_prec(query_res)
    return average_prec


def eval_MAP_QbE(eval_set_path):
    """
    Vyhodnocení - Mean Average Precision pro Query by Example

    :param str eval_set_path: Cesta ke kořenové složce testovacího nebo validačního setu.
    :return: Mean Average Precision pro Query by Example
    """
    data = load_set(eval_set_path)
    results = []

    for i in range(len(data)):
        item = data[i]
        word = item[0]
        correct_sigs = item[1]

        if len(correct_sigs) > 1:  # pokud je jedinny obrazek pro slovo, dotaz je preskocen
            average_prec = __get_average_prec(i, data)
            print(word + ": " + str(average_prec))
            results.append(average_prec)

    mean_average_prec = np.mean(results)
    print("MAP QbE: " + str(mean_average_prec))
    return mean_average_prec
