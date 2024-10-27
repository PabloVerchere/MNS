
import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

N=1000
n1=100
r=5
I=100
T=2
Dt=T/N
h=1/I
lambda_=Dt/(h**2)

def u(x,t):
    return t*np.sin(0.5*np.pi*x)

def alpha(t):
    return 0

def beta(t):
    return t

def f(x,t):
    return np.sin(0.5*np.pi*x)+t*((0.5*np.pi)**2)*np.sin(0.5*np.pi*x)

U=np.zeros((I+1,N+1))

x=np.zeros(I+1)
for i in range(I+1):
   U[i,0]=u(i*h,0)
   x[i]=i*h


A = sp.lil_matrix((I+1, I+1))

A[0,0]=1
A[I,I]=1
for i in range(1,I):
    A[i,i]=1+2*lambda_
    A[i,i+1]=-lambda_
    A[i,i-1]=-lambda_

A = A.tocsc()

b=np.zeros(I+1)

for i in range(1,I):
    b[i]=Dt*f(x[i],0)
b[0]=alpha(0)
b[I]=beta(0)

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
    print('U shape:', U.shape)
    print('S shape:', S.shape)
    print('V shape:', V.shape)

  # Adjust matrix shapes to ensure U (m x n), S (n x n), V (n x n):
    U = U[:, :S.shape[0]]  # Limit U to the number of singular values
    S = np.diag(S)  # Convert S (a vector) into a diagonal matrix
    print('Adjusted U shape:', U.shape)
    print('Adjusted S shape:', S.shape)
    print('Adjusted V shape:', V.shape)

    return U, S, V

W,D,V=decompose_matrix(U)
Wr=W.copy()
#enlever les dernieres colonnes
for i in range(r,n1+1):
    Wr[i,i]=0

A2 = sp.csc_matrix(Wr.T @ A @ Wr)

#faire 2 b un pour Uapr
for k in range(n1+1,N+1):
    b=b+U[:,k-1]
    b[0]=alpha((k+1)*Dt)
    b[I]=beta((k+1)*Dt)
    U[:,k]=spsolve(A,b)
    b=Wr.T @ b
    Uapr[:,k]=Wr @ spsolve(A2,b)
    b = Wr @ b
    for i in range(1,I):
        b[i]=Dt*f(x[i],(k+1)*Dt)

#3 erreurs entre u(vraie valeur), U et Uapr
print("erreur :",la.norm(np.abs(Uapr[:,-1]-U[:,-1])))
