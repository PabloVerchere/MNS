
# Importation des bibliothÃ¨ques
import numpy as np
#from scipy import sparse as sp
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot  as plt

# ParamÃ¨tres
N = 10 # Nombre de pas de temps
n1 = 100 # Nombre de pas de temps avant la dÃ©composition
r = 10 # Rang de la dÃ©composition
T = 1 # Temps final
dt = T /N # Pas de temps
ghost = 2

I = 69 # Nombre de points de l'espace
E = 1.0 # Longueur de l'espace
h = E /(I-5) # Pas d'espace

Kappa = 1.0 # Coefficient de diffusion
# lambda_ = dt /(h **2)


# Fonctions
def u(x, t):
    return np.sin(np.pi * x)

def u0(x):
    return u(x, 0)

def f(x, t):
    return (np.pi **2) * np.sin(np.pi * x)

def alpha(t):
    return u(0, t)

def beta(t):
    return u(E, t)

def R(A, B, U):
    return sp.linalg.spsolve(A, B @ U)


# Initialisation de la matrice A
A = sp.lil_matrix((I, I))

A[0, 0] = 1
A[0, 1] = 126 /11

A[1, 0] = 11 /128
A[1, 1] = 1
A[1, 2] = 11 /128

A[I -2, I -3] = 11 /128
A[I -2, I -2] = 1
A[I -2, I -1] = 11 /128

A[I -1, I -1] = 1
A[I -1, I -2] = 126 /11


for i in range(2, I -2):
    A[i, i] = 1

    A[i, i +1] = 2 /11
    A[i, i -1] = 2 /11

A = A.tocsc()


# Initialisation de la matrice B
B = sp.lil_matrix((I, I))

B[0, 0] = 2077 /157
B[0, 1] = - 2943 /110
B[0, 2] = 574 /44
B[0, 3] = 167 /99
B[0, 4] = - 18 /11
B[0, 5] = 57 /110
B[0, 6] = - 131 /1980

B[1, 0] = 585 /512
B[1, 1] = - 141 /64
B[1, 2] = 459 /512
B[1, 3] = 9 /32
B[1, 4] = - 81 /512
B[1, 5] = 3 /64
B[1, 6] = - 3 /512

B[I -1, I -1] = 2077 /157
B[I -1, I -2] = - 2943 /110
B[I -1, I -3] = 574 /44
B[I -1, I -4] = 167 /99
B[I -1, I -5] = - 18 /11
B[I -1, I -6] = 57 /110
B[I -1, I -7] = - 131 /1980

B[I -2, I -1] = 585 /512
B[I -2, I -2] = - 141 /64
B[I -2, I -3] = 459 /512
B[I -2, I -4] = 9 /32
B[I -2, I -5] = - 81 /512
B[I -2, I -6] = 3 /64
B[I -2, I -7] = - 3 /512


for i in range(2, I -2):
    B[i, i] = - 51 /22

    B[i, i +1] = 12 /11
    B[i, i -1] = 12 /11

    B[i, i +2] = 3 /44
    B[i, i -2] = 3 /44


B = B /(Kappa * (h **2))


x = np.linspace(0-2*h, E+2*h, I)
U = np.zeros((I, N))
U[:, 0] = u0(x)

F = f(x, 0)

# -B @ U = A @ F
U[:, 1] = spsolve(-B, A @ F)





fantome = ghost
def rectification(U, dt, x):
    U = U + alpha(dt) - U[fantome] + (beta(dt) - U[-fantome -1] - (alpha(dt) - U[fantome])) * x
    return U





u_exact_sol = np.zeros((I, N))
u_exact_sol = u(x, dt)


print("erreur:", np.linalg.norm(U[:, 1] - u_exact_sol, np.inf))

# rectification en 0 et E
rect_0 = U[ghost, 1] + ((U[-ghost -1, 1] - U[ghost, 1])) * x
rect_val = alpha(dt) + ((beta(dt) - alpha(dt))) * x

#print("rect_0:", rect_0)
#print("rect_val:", rect_val)

Ubis = U[:,1] - rect_0 + rect_val

Uter = rectification(U[:, 1], dt, x)

#Ubis = Usec+alpha(0)-Usec[2]+(beta(0)-Usec[-3]-(alpha(0)-Usec[2]))*x 

#Ubis = U[:, 1]-U[2, 1]-(U[-3, 1]-U[2, 1])*x

print("erreurbis:", np.linalg.norm(Ubis - u_exact_sol, np.inf))
print("erreurter:", np.linalg.norm(Uter - u_exact_sol, np.inf))