import numpy as np
import pandas as pd
import utile as utl
import acp.ACP as acp
import aef.AEF as aef
import factor_analyzer as fa
import grafice as g


# Citirea datelor despre consumul de alcool din fisierul csv
tabel = pd.read_csv('dataIN/gapminder_alcohol.csv', index_col=0, na_values=' ')
# print(tabel)

matrice_numerica = tabel.values

# Inlocuirea valorilor inexistente/nule cu media
X = utl.inlocuireNAN(matrice_numerica)
# print(X)

# Citirea tarilor lumii cu maparea lor pe continente
tari_continente = pd.read_csv('dataIN/country_continent.csv', index_col=0, na_values=' ')
# print(continente_tari)

# Join intre DataFrame cu consumul de alcool pe tari si cel de continente-tari
tabel1 = pd.merge(left=tabel, right=tari_continente, on='Country')
# print(tabel1)

# Pastram doar datele pentru Europa
continent = ['Europe']
rezultat_df = tabel1.loc[tabel1['Continent'].isin(continent)]
# Stergem coloana continent
rezultat_df.pop('Continent')

# Salvam in fisier csv datele pe care le vom analiza in continuare
rezultat_df.to_csv('dataOUT/DateEuropa.csv')

aefModel = aef.AEF(rezultat_df)
Xstd = aefModel.getXstd()

# Salvarea matricei standardizate in fisier csv
obsNume = rezultat_df.index.values
varNume = rezultat_df.columns.values
Xstd_df = pd.DataFrame(data=Xstd, index=obsNume, columns=varNume)
Xstd_df.to_csv('dataOUT/Xstd.csv')

# calcul test de sfericitate Bartlett
sfericitateBartlett = fa.calculate_bartlett_sphericity(Xstd_df)  # ia ca parametru un DF cu valori standardizate
print(sfericitateBartlett)
if sfericitateBartlett[0] > sfericitateBartlett[1]:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori comuni!')
    exit(-1)

# Calcul indici Kaiser-Meyer-Olkin (KMO)
kmo = fa.calculate_kmo(Xstd_df)  # ia ca parametru un DF cu valori standardizate
print(kmo)
vector = kmo[0]
print(type(vector)); print(vector.shape)
matrice = vector[:, np.newaxis]
print(matrice); print(matrice.shape)
matrice_df = pd.DataFrame(data=matrice,
                          columns=['Indici_KMO'],
                          index=varNume)
g.corelograma(matrice=matrice_df,
              dec=5,
              titlu='Corelograma indicilor Kaiser-Meyer-Olkin')
# g.afisare()

if kmo[1] >= 0.5:
    print('Exista cel putin un factor comun!')
else:
    print('Nu exista factori comuni!')
    exit(-2)

# Extragere factori semnificativi
nrFactoriSemnificativi = 1
chi2TabMin = 1
for k in range(1, varNume.shape[0]):
    faModel = fa.FactorAnalyzer(n_factors=k)
    faModel.fit(X=Xstd_df)
    factoriComuni = faModel.loadings_  # factorii comuni - factorii de corelatie
    print(factoriComuni)
    factoriSpecifici = faModel.get_uniquenesses()
    print(factoriSpecifici)

    chi2Calc, chi2Tab = aefModel.calculTestBartlett(factoriComuni, factoriSpecifici)
    print(chi2Calc, chi2Tab)
    aefModel.calculTestBartlett(factoriComuni, factoriSpecifici)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break
    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        nrFactoriSemnificativi = k

print('Numar factori semnificativi extrasi: ', nrFactoriSemnificativi)


acpModel = acp.ACP(rezultat_df)
Xstd2 = acpModel.getXstd()
Xstd2_df = pd.DataFrame(data=Xstd2, index=obsNume, columns=varNume)
Xstd2_df.to_csv('dataOUT/Xstd2.csv')


# Matricea de corelații
R = acpModel.getCorr()
numeInstante = rezultat_df.index
numeVariabile = rezultat_df.columns
nrVariabile = len(numeVariabile)
tabelCorelatii = pd.DataFrame(R, index=numeVariabile, columns=numeVariabile)
tabelCorelatii.to_csv("dataOUT/MatriceCorelatii.csv")

R_fact = acpModel.getRxc()
tabelCorelatiiFactori = utl.tabelare(R_fact, numeColoane=["Componenta " + str(i) for i in range(1, nrVariabile + 1)],
                                        numeInstante=numeVariabile, tabel="dataOUT/CorelatiiFactoriale.csv")


# Varianța componentelor
alpha = acpModel.getValProp();
tabelVarianta = utl.tabelareVarianta(alpha)
tabelVarianta.to_csv("dataOUT/VariantaComponentelor.csv")

componente = acpModel.getCompPrin()
scoruri = acpModel.getScoruri()
q = acpModel.getCalObs()
beta = acpModel.getBetha()
comun = acpModel.getComun()


# Cosinusuri și contribuții
utl.tabelare(q,
             numeColoane=["Componenta " + str(i) for i in range(1, nrVariabile + 1)],
             numeInstante=numeInstante,
             tabel="dataOUT/Cosinusuri.csv")

utl.tabelare(beta,
             numeColoane=["Componenta " + str(i) for i in range(1, nrVariabile + 1)],
             numeInstante=numeInstante,
             tabel="dataOUT/Contributii.csv")


# Comunalități
comunalitati = utl.tabelare(comun,
                            numeColoane=["Componenta " + str(i) for i in range(1, nrVariabile + 1)],
                            numeInstante=numeVariabile,
                            tabel="dataOUT/Comunalitati.csv")


### Grafice

g.corelograma(matrice=tabelCorelatiiFactori,
              titlu='Corelograma factorilor de corelatie')
# g.afisare()


# Cercul corelatiilor variabilelor observate
g.cerculCorelatiilor(matrice=tabelCorelatiiFactori,
                     titlu='Cercul corelatiilor variabilelor observate')
# g.afisare()


ACPvaloriProprii = acpModel.getValProp()
# grafic valori proprii din ACP
g.componentePrincipale(valoriProprii=ACPvaloriProprii,
                       titlu='Varianta explicata de componentele principale din ACP')
# g.afisare()


# Componente si scoruri
g.norPuncte(matrice=componente, titlu='Grafic componente')
g.norPuncte(matrice=scoruri, titlu='Grafic scoruri')
# g.afisare()

# Comunalitati
g.corelograma(matrice=comunalitati, titlu='Corelograma comunalitati')
g.afisare()
