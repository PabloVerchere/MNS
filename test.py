from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

myimg=Image.open("C:\projet 4A\polytech.jpg")
plt.imshow(myimg) ; plt.show()


R, G, B = myimg.split()
red=np.asarray( R )
green=np.asarray( G )
blue=np.asarray( B )

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

def reconstruct_matrix(U, S, V, r):
  '''Reconstructs the matrix using the SVD components with rank r.'''
  print('\nReconstructing matrix with rank=', r)
  R = U[:, :r] @ S[:r, :r] @ V[:r, :]  # Use rank-reduced components for approximation
  print(R)
  print('R shape:', R.shape)  # Output shape of the approximated matrix
  return R

U1, S1, V1 = decompose_matrix(red)  # Decompose with SVD
U2, S2, V2 = decompose_matrix(green)  # Decompose with SVD
U3, S3, V3 = decompose_matrix(blue)  # Decompose with SVD

red2=reconstruct_matrix(U1, S1, V1, 10)
green2=reconstruct_matrix(U2, S2, V2, 10)
blue2=reconstruct_matrix(U3, S3, V3, 10)
R2=Image.fromarray(red2).convert('L')
G2=Image.fromarray(green2).convert('L')
B2=Image.fromarray(blue2).convert('L')

img2 = Image.merge( 'RGB', (R2, G2, B2))
plt.imshow(img2) ; plt.show()
