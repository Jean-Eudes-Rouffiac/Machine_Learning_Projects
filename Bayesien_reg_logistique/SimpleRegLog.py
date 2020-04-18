
# coding: utf-8

# # Regression logistique
# Application sur les donnees Iris

from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

plt.close("all")

#%% ==== Trace de frontiere de decision en 2D
def plot_regions_decision_2d(X, y, classifier, resolution=0.02, titre=' '):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0, X[:, 0].max() + 0
    x2_min, x2_max = X[:, 1].min() - 0, X[:, 1].max() + 0
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.6, c=cmap(idx),
                    marker=markers[idx], label= 'classe {}'.format(cl))
    plt.legend(loc='best')
    plt.title(titre, fontsize=12)
    
#%% ### Dataset : IRIS (inclus dans Sklearn)
iris = datasets.load_iris()
X, Y = iris.data, iris.target

#%% ### Decoupage des donnees en jeu d'apprentissage et de validation
# Remarque : la normalisation des donnees est omise dans ce exemple

Xa, Xv, Ya, Yv = train_test_split(X, Y, shuffle=True, test_size=1/2, stratify=Y)


#%% ### Selection de l'hyper-parametre C 

clf_reglog = linear_model.LogisticRegression(tol=1e-5, multi_class='multinomial', solver='lbfgs')
vectC = np.logspace(-3, 2, 15)
err_app = np.empty(vectC.shape[0])
err_val = np.empty(vectC.shape[0])
for ind_C, C in enumerate(vectC):
    clf_reglog.C = C
    clf_reglog.fit(Xa, Ya)
    err_val[ind_C] = 1 - accuracy_score(Yv, clf_reglog.predict(Xv))
    err_app[ind_C] = 1 - accuracy_score(Ya, clf_reglog.predict(Xa))

# Choix du meilleur C
err_min_val, ind_min = err_val.min(), err_val.argmin()
Copt = vectC[ind_min]


#%% #### Trace des courbes d'erreur d'apprentissage
plt.figure()
plt.semilogx(vectC, err_val, color='green', linestyle='--', marker='s', markersize=5, label='Validation')
plt.semilogx(vectC, err_app, color='blue', linestyle='--', marker='s', markersize=5, label='Apprentissage')
plt.xlabel('Parametre C')
plt.ylabel('Erreur Classification')
plt.legend(loc='best')

#%% ##### Model final
clf_reglog.C = Copt
clf_reglog.fit(Xa, Ya)
print('Valeur de Copt = {}'.format(Copt ))
print('Err validation correspondante = {}'.format(100*(1 - accuracy_score(Yv, clf_reglog.predict(Xv)))))
print('Err apprentissage correspondante = {}'.format(100*(1 - accuracy_score(Ya, clf_reglog.predict(Xa)))))


#%% ### Trace de la frontiere de decision en 2D
# Utilisation de 2 variables choisies parmi les 4.
# trace frontiere de decision en fonction de certaines variables choisies
#variables = [2, 3]
## on re-apprend le modele avec Coptimal 
#clf_reglog.C = Copt
#clf_reglog.fit(Xa[:,variables], Ya) 
#plot_regions_decision_2d(Xa[:,variables], Ya, clf, 0.02, titre='{}'.format("Regression Logistique"))
#

