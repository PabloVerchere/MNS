# Importation des bibliothèques
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pyplot as plt

import data

import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath('.'))
import image_compression.src.fct as fct


start_time = time.time()


# Initialisation de x et U
x = np.linspace(0, data.E, data.I +1)
U = np.zeros((data.I +1, data.N +1))
U[:, 0] = data.u0(x)


# Initialisation de la matrice A
A = sp.lil_matrix((data.I +1, data.I +1))

A[0, 0] = 1
A[data.I, data.I] = 1

for i in range(1, data.I):
    A[i, i] = 1 + (2 * data.lambda_)
    A[i, i +1] = -data.lambda_
    A[i, i -1] = -data.lambda_

A = A.tocsc()

# Initialisation du membre de droite
b = np.zeros(data.I +1)
b[1: data.I] = data.dt * data.f(x[1: data.I], 0)
b[0] = data.alpha(0)
b[data.I] = data.beta(0)

# A @ U[k+1] = U[k] + dt * f(x, k)
# Résolution du système linéaire jusqu'au temps n1
for k in range(1, data.n1 +1):
    b = b + U[:, k -1]

    b[0] = data.alpha((k +1) * data.dt)
    b[data.I] = data.beta((k +1) * data.dt)

    U[:, k] = spsolve(A, b)

    b[1: data.I] = data.dt * data.f(x[1: data.I], (k +1) * data.dt)
    

Uapr = U.copy()



W, D, V = fct.decompose_matrix(U)
Wr = W[:, :data.r].copy() # On garde les r premières colonnes de W

Ar = sp.csc_matrix(Wr.T @ A @ Wr)
bapr = b.copy()


for k in range(data.n1 +1, data.N +1):
    b = b + U[:, k -1]

    b[0] = data.alpha((k +1) * data.dt)
    b[data.I] = data.beta((k +1) * data.dt)

    U[:, k] = spsolve(A, b)

    b[1: data.I] = data.dt * data.f(x[1: data.I], (k +1) * data.dt)



    bapr = bapr + Uapr[:, k -1]

    bapr[0] = data.alpha((k +1) * data.dt)
    bapr[data.I] = data.beta((k +1) * data.dt)

    bapr = Wr.T @ bapr
    Uapr[:, k] = Wr @ spsolve(Ar, bapr)
    bapr = Wr @ bapr
    
    bapr[1: data.I] = data.dt * data.f(x[1: data.I], (k +1) * data.dt)



# Affichage des erreurs finales
print("Erreur entre la solution exacte et la solution approchée classique: ", la.norm(data.u(x, data.T) - U[:, -1]))
print("Erreur entre la solution exacte et la solution approchée par POD: ", la.norm(data.u(x, data.T) - Uapr[:, -1]))
print("Erreur entre les 2 solutions approchées: ", la.norm(U[:, -1] - Uapr[:, -1]))
print("Temps d'exécution : %.2f secondes" %(time.time() - start_time))
print("Rang de la projection:", data.r)


# Affichage des erreurs en fonction des itérations
erreur_classique = []
erreur_pod = []
erreur_2 = []

for k in range(data.N +1):
    erreur_classique.append(la.norm(data.u(x, k * data.dt) - U[:, k]))
    erreur_pod.append(la.norm(data.u(x, k * data.dt) - Uapr[:, k]))
    erreur_2.append(la.norm(U[:, k] - Uapr[:, k]))

# Plot des erreurs
plt.plot(range(data.N +1), erreur_classique, label ="Erreur classique")
plt.plot(range(data.N +1), erreur_pod, label ="Erreur POD")
plt.plot(range(data.N +1), erreur_2, label ="Erreur entre les 2 approches")

plt.axvline(x =data.n1, color ='r', linestyle ='--', label ='n1')
plt.xlabel("Itérations")
plt.ylabel("Erreur")
plt.title("Erreurs en fonction des itérations")

plt.legend()
plt.grid(True)
plt.show()