import numpy as np

# Paramètres
N = 1000 # Nombre de pas de temps
n1 = 100 # Nombre de pas de temps avant la décomposition
r = 10 # Rang de la décomposition
T = 2.0 # Temps final
dt = T /N # Pas de temps

I = 5 # Nombre de points de l'espace
E = 1.0 # Longueur de l'espace
h = E /I # Pas d'espace

K = 1.0 # Coefficient de diffusion
lambda_ = dt /(h **2)


# Fonctions
def u(x, t):
    return np.exp(x + t)

def u0(x):
    return u(x, 0)

def f(x, t): # f(x, t) = (du / dt) - (K * d2u / dx2)
    return (1 - K) * np.exp(x + t)
    
def alpha(t):
    return u(0, t)

def beta(t):
    return u(E, t)