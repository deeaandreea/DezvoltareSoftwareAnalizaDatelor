import sklearn.cross_decomposition as skl
import pandas as pd
import numpy as np
import utils as utl
import visual as vi

# Citim datele din acelasi fisier csv, in care avem doua seturi de date
input_file = "./dataIN/CitiesData.csv"
tabel = pd.read_csv(input_file, index_col=0)
print(tabel)

# Eliminam simbolul % din coloanele cu valori procentuale
tabel['Remote Jobs']=tabel['Remote Jobs'].replace('%','',regex=True).astype(float)
tabel['Overworked Population']=tabel['Overworked Population'].replace('%','',regex=True).astype(float)
tabel['Inflation']=tabel['Inflation'].replace('%','',regex=True).astype(float)


var_nume = tabel.columns[1:].values
print(var_nume)
obs_nume = tabel.index.values
print(obs_nume)

# Stabilim parametrii modelului
# Coloanele matricii X cu p variabile cantitative
x_var = var_nume[:4]
print(x_var)
# Coloanele matricii Y cu q variabile cantitative
y_var = var_nume[4:8]
print(y_var)

X = tabel[x_var].values
print(X)
Xstd = utl.standardize(X)
Xstd_df = pd.DataFrame(data=Xstd, index=obs_nume,
                    columns=x_var)
Xstd_df.to_csv('./dataOUT/Xstd.csv')

Y = tabel[y_var].values
Ystd = utl.standardize(Y)
Ystd_df = pd.DataFrame(data=Ystd, index=obs_nume,
                    columns=y_var)
Ystd_df.to_csv('./dataOUT/Ystd.csv')

n, p = np.shape(X)
print(n, p)
q = np.shape(Y)[1]
print(q)
m = min(p, q)  # numarul de perechi canonice
print(m)

# Modelul primeste ca parametru numarul de perechi canonice m
accModel = skl.CCA(n_components=m)
accModel.fit(X=Xstd, Y=Ystd)

# Extragere radacini canonice
z, u = accModel.transform(X=Xstd, Y=Ystd)  # returneaza x_scores si y_scores
print(z)
z = np.fliplr(z)
z_df = pd.DataFrame(data=z, index=obs_nume,
                    columns=['z'+str(j+1) for j in range(p)])
z_df.to_csv('./dataOUT/z.csv')
vi.corelograma(matrice=z_df, titlu='Variabile canonice z')
# vi.afisare()

u = np.fliplr(u)
u_df = pd.DataFrame(data=u, index=obs_nume,
                    columns=['u'+str(j+1) for j in range(q)])
u_df.to_csv('./dataOUT/u.csv')
vi.corelograma(matrice=u_df, titlu='Variabile canonice u')
# vi.afisare()

# Extragere factori de corelatie (factor loadings)
# Corelatia dintre variabilele cauzale X si variabilele canonice z
Rxz = accModel.x_loadings_
Rxz_df = pd.DataFrame(data=Rxz, index=x_var,
                      columns=['z'+str(j+1) for j in range(m)])
Rxz_df.to_csv('./dataOUT/Rxz.csv')
vi.corelograma(matrice=Rxz_df,
               titlu='Corelatia dintre variabilele cauzale X si variabilele canonice z')
# vi.afisare()

# Corelatia dintre variabilele cauzale Y si variabilele canonice u
Ryu = accModel.y_loadings_
Ryu_df = pd.DataFrame(data=Ryu, index=y_var,
                      columns=['u'+str(j+1) for j in range(m)])
Ryu_df.to_csv('./dataOUT/Ryu.csv')
vi.corelograma(matrice=Ryu_df,
               titlu='Corelatia dintre variabilele cauzale Y si variabilele canonice u')
vi.afisare()
