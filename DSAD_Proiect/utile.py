import numpy as np
import pandas as pd


def inlocuireNAN(X):
    medii = np.nanmean(X, axis=0)  # calcul medii pentru coloanele cu valori NAN
    poz = np.where(np.isnan(X))
    print('Localizare NaN:', poz)
    X[poz] = medii[poz[1]]
    return X


def tabelare(X, numeColoane=None, numeInstante=None, tabel=None):
    Xtabel = pd.DataFrame(X)
    if numeColoane is not None:
        Xtabel.columns = numeColoane
    if numeInstante is not None:
        Xtabel.index = numeInstante
    if tabel is None:
        Xtabel.to_csv("TabelDefault.csv")
    else:
        Xtabel.to_csv(tabel)
    return Xtabel


def tabelareVarianta(alpha):
    m = len(alpha)
    variantaCumulata = np.cumsum(alpha)
    procentVarianta = alpha * 100 / m
    procentCumulat = np.cumsum(procentVarianta)
    tabelVarianta = pd.DataFrame(data={
        "Varianta": alpha,
        "Varianta cumulata": variantaCumulata,
        "Procent varianta": procentVarianta,
        "Procent cumulat": procentCumulat
    })
    return tabelVarianta
