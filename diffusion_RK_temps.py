# Importation des bibliothÃƒÂ¨ques
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot  as plt


# ParamÃƒÂ¨tres
N = 10000 # Nombre de pas de temps
n1 = 20 # Nombre de pas de temps avant la dÃƒÂ©composition
r = 10 # Rang de la dÃƒÂ©composition
T = 1 # Temps final
dt = T /(N -1) # Pas de temps
fantome = 2 # Nombre de points fantÃƒÂ´mes

more = 1 + 2 *fantome
I = 16 + more # Nombre de points de l'espace
E = 1.0 # Longueur de l'espace
h = E /(I - more) # Pas d'espace

Kappa = 1.0 # Coefficient de diffusion
# lambda_ = dt /(h **2)


# Fonctions
def u(x, t):
    return np.sin(np.pi * x) * np.exp(-(np.pi **2) * t)

def u0(x):
    return u(x, 0)

def f(x, t): # nulle
    return x * 0

def alpha(t):
    return u(0, t) # = 0

def beta(t):
    return u(E, t) # = 0

def R(A, B, U):
    return spsolve(A, B @ U)

def R2(A, B, U):
    return Wr @ spsolve(A, B @ (Wr.T @ U))

def rectification(U, dt, x):
    U = U + alpha(dt) - U[fantome] + (beta(dt) - U[-fantome -1] - (alpha(dt) - U[fantome])) * x
    return U


def RK4(Un, A, B, x, k):
    k0 = dt * rectification(R(A, B, Un), k * dt, x)
    U1 = Un + (k0 * 0.5)

    k1 = dt * rectification(R(A, B, U1), k * dt, x)
    U2 = Un + (k1 * 0.5)

    k2 = dt * rectification(R(A, B, U2), k * dt, x)
    U3 = Un + k2

    k3= dt * rectification(R(A, B, U3), k * dt, x)
    K = k0 + (2 * k1) + (2 * k2) + k3

    return Un + (K /6)

def RK4_2(Un, A, B, x, k):
    k0 = dt * rectification(R2(A, B, Un), k * dt, x)
    U1 = Un + (k0 * 0.5)

    k1 = dt * rectification(R2(A, B, U1), k * dt, x)
    U2 = Un + (k1 * 0.5)

    k2 = dt * rectification(R2(A, B, U2), k * dt, x)
    U3 = Un + k2

    k3= dt * rectification(R2(A, B, U3), k * dt, x)
    K = k0 + (2 * k1) + (2 * k2) + k3

    return Un + (K /6)


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


x = np.linspace(0 - (fantome * h), E + (fantome * h), I)
U = np.zeros((I, N))
u_exact_sol = np.zeros((I, N))



U[:, 0] = u0(x)
u_exact_sol[:, 0] = u(x, 0)

F = f(x, 0)

for k in range(1,n1 +1):
    u_exact_sol[:, k] = u(x , k * dt)

    U[:, k] = RK4(U[:, k-1], A, B, x, k-1)

    # Rectification des poitns fantÃƒÂ´mes
    U[0, k] = -U[2 * fantome, k]
    U[1, k] = -U[2 * fantome -1, k]
    U[-1, k] = -U[-2 * fantome -1, k]
    U[-2, k] = -U[-2 * fantome, k]





"Decomposes the matrix M using Singular Value Decomposition (SVD)"
def decompose_matrix(M):
    U, S, V = np.linalg.svd(M) # Perform SVD
    S = np.diag(S)  # Convert S (a vector) into a diagonal matrix of dim m x m

    return U, S, V

Uapr = U.copy()
W, D, V = decompose_matrix(U)
Wr = W[:, :r].copy()

A2 = sp.csc_matrix(Wr.T @ A @ Wr)
B2 = sp.csc_matrix(Wr.T @ B @ Wr)




for k in range(n1 +1, N -1):
    u_exact_sol[:, k] = u(x , k * dt)


    U[:, k ] = RK4(U[:, k-1], A, B, x, k-1)
    Uapr[:, k ] = RK4_2(Uapr[:, k-1], A2, B2, x, k-1)
    
    # Rectification des poitns fantÃƒÂ´mes
    U[0, k] = -U[2 * fantome, k]
    U[1, k] = -U[2 * fantome -1, k]
    U[-1, k] = -U[-2 * fantome -1, k]
    U[-2, k] = -U[-2 * fantome, k]

    Uapr[0, k] = -Uapr[2 * fantome, k]
    Uapr[1, k] = -Uapr[2 * fantome -1, k]
    Uapr[-1, k] = -Uapr[-2 * fantome -1, k]
    Uapr[-2, k] = -Uapr[-2 * fantome, k]




# Erreur au terme d'indice k
k=101
print("exacte :", u_exact_sol[:, k])
print("classique :", U[:, k])
print("approchee :", Uapr[:, k])

print("Erreur entre classique et rÃ©elle: ", la.norm(np.abs(U[:, k] - u_exact_sol[:, k]), np.inf))
print("Erreur entre approchÃ©e et rÃ©elle: ", la.norm(np.abs(Uapr[:, k] - u_exact_sol[:, k]), np.inf))
print("Erreur entre classique et approchÃ©e: ", la.norm(np.abs(Uapr[:, k] - U[:, k]), np.inf))

plt.plot(x, u_exact_sol[:, k],
         x, U[:, k],
         x, Uapr[:, k])
plt.legend(["Exacte", "Classique", "ApprochÃ©e"])
plt.show()