# Importation des bibliothèques
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Paramètres
N = 1000 # Nombre de pas de temps
n1 = 100 # Nombre de pas de temps pour la décomposition
r = 10 # Rang de la décomposition
T = 2.0 # Temps final
Dt = T /N # Pas de temps

I = 100 # Nombre de points de l'espace
E = 1.0 # Longueur de l'espace
h = E /I # Pas d'espace

K = 1.0 # Coefficient de diffusion
lambda_ = Dt /(h **2)


# Fonctions
def u(x, t):
    return np.cos(np.pi * x * t)

def u0(x):
    return u(x, 0)

def f(x, t): # f(x, t) = (du / dt) - (K * d2u / dx2)
    return -((np.pi * x) * (np.sin(np.pi * x * t))) + (K * ((np.pi * t) **2) * np.cos(np.pi * x * t))
    
def alpha(t):
    return u(0, t)

def beta(t):
    return u(E, t)


U=np.zeros((I+1,N+1))

x=np.zeros(I+1)

for i in range(I+1):
   U[i,0]=u(i*h,0)
   x[i]=i*h

# Init A matrix
A = sp.lil_matrix((I+1, I+1))

A[0,0]=1
A[I,I]=1
for i in range(1,I):
    A[i,i]=1+2*lambda_
    A[i,i+1]=-lambda_
    A[i,i-1]=-lambda_

A = A.tocsc()


# Init b vector
b=np.zeros(I+1)

for i in range(1,I):
    b[i]=Dt*f(x[i],0)
b[0]=alpha(0)
b[I]=beta(0)

# Resolution
for k in range(1,n1+1):
    b=b+U[:,k-1]
    b[0]=alpha((k+1)*Dt)
    b[I]=beta((k+1)*Dt)

    U[:,k]=spsolve(A,b)
    for i in range(1,I):
        b[i]=Dt*f(x[i],(k+1)*Dt)

Uapr=U.copy()


def decompose_matrix(M):
    '''Decomposes the matrix M using Singular Value Decomposition (SVD).'''
    U, S, V = np.linalg.svd(M)  # Perform SVD

  # Adjust matrix shapes to ensure U (m x n), S (n x n), V (n x n):
    U = U[:, :S.shape[0]]  # Limit U to the number of singular values
    S = np.diag(S)  # Convert S (a vector) into a diagonal matrix

    return U, S, V



W,D,V=decompose_matrix(U)
Wr=W[:,:r].copy()

A2 = sp.csc_matrix(Wr.T @ A @ Wr)

bapr=b.copy()
b2=bapr.copy()

# Resolution avec la décomposition
for k in range(n1+1,N+1):
    # ajout des u d'indice k et des termes de bords pour le calcul en k+1
    b=b+U[:,k-1]
    b[0]=alpha((k+1)*Dt)
    b[I]=beta((k+1)*Dt)

    bapr=bapr+Uapr[:,k-1]
    bapr[0]=alpha((k+1)*Dt)
    bapr[I]=beta((k+1)*Dt)
    
    # resolution pour U avec Euler Implicite classique
    U[:,k]=spsolve(A,b)

    # resolution avec l'approximation de rang r
    b2=Wr.T @ bapr
    Uapr[:,k]=Wr @ spsolve(A2, b2)

    # debut de la construction des 2 seconds membres
    for i in range(1,I):
        b[i]=Dt*f(x[i],(k+1)*Dt)
        bapr[i]=Dt*f(x[i],(k+1)*Dt)



#3 erreurs entre u(vraie valeur), U et Uapr
print("Erreur entre res calssique et val réelle :",la.norm(np.abs(U[:,-1]-u(x,T))))
print("Erreur entre res approchée et val réelle :",la.norm(np.abs(Uapr[:,-1]-u(x,T))))
print("Erreur entre res calssique et approchée :",la.norm(np.abs(Uapr[:,-1]-U[:,-1])))
