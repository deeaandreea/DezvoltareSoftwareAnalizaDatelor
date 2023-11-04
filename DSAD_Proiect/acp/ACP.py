'''
Clasa care incapsuleaza implementarea ACP
'''
import numpy as np


class ACP:

    def __init__(self, matrice):
        self.X = matrice

        # calcul matrice corelatie pentru X nestandardizat
        self.R = np.corrcoef(self.X, rowvar=False)  # avem variabilele pe coloane

        # standardizare valori X
        medii = np.mean(self.X, axis=0)  # variabilele se gasesc pe coloane
        abateri = np.std(self.X, axis=0)  # calcul pe coloane
        self.Xstd = (self.X - medii) / abateri

        # calcul matrice varianta/covarianta pentru X standardizat
        self.Cov = np.cov(self.Xstd, rowvar=False)  # avem variabilele pe coloane

        # calcul valori proprii si vectori proprii pentru matricea de varianta/covarianta
        valProp, vectProp = np.linalg.eigh(self.Cov)
        print(valProp)

        # sortare descrescatoare a valorilor proprii si vectorilor proprii
        k_des = [k for k in reversed(np.argsort(valProp))]
        print(k_des)
        self.alpha = valProp[k_des]
        self.a = vectProp[:, k_des]
        print(self.alpha)

        # regularizare vectorilor proprii
        for col in range(self.a.shape[1]):
            minim = np.min(self.a[:, col])  # calcul mim si max pe coloane, pentru fiecare vector propriu
            maxim = np.max(self.a[:, col])
            if np.abs(minim) > np.abs(maxim):
                # self.a[:, col] = -self.a[:, col]
                self.a[:, col] *= -1

        # calcul componente principale
        self.C = self.Xstd @ self.a  # operatorul @ este supraincarcat pentru inmultire matriceala

        # calcul matrice factori de corelatie (factor loadings)
        # reprezinta corelatia intre variabilele initiale si componentele principale
        self.Rxc = self.a * np.sqrt(self.alpha)

        # calcul scoruri (componetele principale standardizate)
        self.scoruri = self.C / np.sqrt(self.alpha)

        # calcul calitatii reprezentarii observatilor pe axele componentelor principale
        C2 = self.C * self.C
        C2sum = np.sum(C2, axis=1)  # sume pe linii, pentru fiecare observatie
        self.CalObs = np.transpose(np.transpose(C2) / C2sum)

        # contributia observatiilor la varianta componentelor principale
        self.betha = C2 / (self.alpha * self.X.shape[0])

        # calcul comunitati (regasirea componetelor principale in variabilele initiale)
        Rxc2 = self.Rxc * self.Rxc
        self.Comun = np.cumsum(Rxc2, axis=1)  # sume cumulative pe linii, pentru fiecare variabila observata


    def getCorr(self):
        return self.R

    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.alpha

    def getCompPrin(self):
        return self.C

    def getRxc(self):
        return self.Rxc

    def getScoruri(self):
        return self.scoruri

    def getCalObs(self):
        return self.CalObs

    def getBetha(self):
        return self.betha

    def getComun(self):
        return self.Comun