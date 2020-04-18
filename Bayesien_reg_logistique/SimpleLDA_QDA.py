
# coding: utf-8

#%% =========== Illustration de LDA et QDA ====
# Application sur les donnees Iris
# =============================================
#from __future__ import print_function
#from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

plt.close('all')

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
    
# %% ### Dataset : IRIS (inclus dans Sklearn)
# Caractéristiques : 150 points, 4 variables, 3 classes
# chargement des donnees
iris = datasets.load_iris()
X, Y = iris.data, iris.target
print('Nombre de points : {}'.format(X.shape[0]))
print('Nombre de variables : {}'.format(X.shape[1]))
print('Nombre de classes : {}'.format(len(np.unique(Y))))
classes, nbpoints = np.unique(Y, return_counts=True)
for i, lab in enumerate(classes):
    print('Classe {} comprend {} points'.format(lab, nbpoints[i]))


#%% ### Utilisation de LDA et QDA
# Remarque : la normalisation des donnees, le decoupage apprentissage/validation/test 
# sont omis dans ce exemple

# LDA
clf_lda = LinearDiscriminantAnalysis(solver='svd', store_covariance = True)
clf_lda.fit(X, Y)
Y_lda = clf_lda.predict(X)
err_lda = sum(Y_lda != Y)/Y.size
print('LDA : taux d''erreur = {}%'.format(100*err_lda))

# QDA
clf_qda = QuadraticDiscriminantAnalysis(store_covariance = True)
clf_qda.fit(X, Y)
print(clf_qda.means_)
Y_qda = clf_qda.predict(X)
err_qda = sum(Y_qda!= Y)/Y.size
print('QDA : taux d''erreur = {}%'.format(100*err_qda))


#%% ### Trace de la frontiere de decision en 2D
# Utilisation de 2 variables choisies parmi les 4. 
#On fait le modele LDA et QDA que pour ces variables

#variables = [2, 3]
## on ne fait ceci que pour le trace de la frontiere de decision de la LDA et QDA en 2D
#classifieur = 'LDA'
#clf_lda.fit(X[:,variables], Y) 
#plot_regions_decision_2d(X[:,variables], Y, clf_lda, 0.02, titre='LDA')
#
#clf_qda.fit(X[:,variables], Y) 
#plot_regions_decision_2d(X[:,variables], Y, clf_qda, 0.02, titre='QDA')

