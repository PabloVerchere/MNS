from PIL import Image
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import shutil

import data



"Decomposes the matrix M using Singular Value Decomposition (SVD)"
def decompose_matrix(M):
  U, S, V = np.linalg.svd(M) # Perform SVD
  S = np.diag(S)  # Convert S (a vector) into a diagonal matrix

  return U, S, V


"Reconstructs the matrix using the SVD components with rank r"
def reconstruct_matrix(U, S, V, r):
  R = U[:, :r] @ S[:r, :r] @ V[:r, :]  # Use rank-reduced components for approximation

  return R


"SVD compression"
def svd(img, rank):
    print("Compressing image with SVD...")

    img_rank = []
    
    # Color channels separation
    R, G, B = img.split()
    red = np.asarray(R)
    green = np.asarray(G)
    blue = np.asarray(B)

    # Decompose with SVD
    Ur, Sr, Vr = decompose_matrix(red)
    Ug, Sg, Vg = decompose_matrix(green)
    Ub, Sb, Vb = decompose_matrix(blue)

    for r in rank:
        print("   Reconstructing matrix with rank = " + str(r))
        red_rank = reconstruct_matrix(Ur, Sr, Vr, r)
        green_rank = reconstruct_matrix(Ug, Sg, Vg, r)
        blue_rank = reconstruct_matrix(Ub, Sb, Vb, r)

        # Convert numpy arrays to PIL images
        R_rank = Image.fromarray(red_rank).convert('L')
        G_rank = Image.fromarray(green_rank).convert('L')
        B_rank = Image.fromarray(blue_rank).convert('L')

        img_rank.append(Image.merge("RGB", (R_rank, G_rank, B_rank)))
        
    print()
    return img_rank


"Returns the size of the image in a human-readable format"
def size_formating(size_b):
    # Convert bytes to kb and Mb
    size_kb = size_b / 1024
    size_mb = size_kb / 1024

    # Choose the appropriate format to return
    if size_mb >= 1:
        return (str(round(size_mb, 2)) + " Mb")
    else:
        return (str(round(size_kb, 2)) + " kb")


"Updates the full name and absolute path of the image"
def update_full_name():
    for r in data.rank:
        data.full_name_rank.append(data.file_name + "_rank_" + str(r) + data.extension)

    data.full_name = data.file_name + data.extension
    data.absolute_path = os.path.abspath(data.src_img + data.full_name)


def clean_reduced_file():
    print("Cleaning compressed images...")

    img_dir = os.path.join("img")

    for filename in os.listdir(img_dir):
        if filename.endswith("_rank_reduced"):
            shutil.rmtree(os.path.join(img_dir, filename)) # Remove the directory and its content


def saveIMG(img_rank):
    print("Saving compressed images...")

    # Create the directory to save the compressed images
    output_dir = os.path.join("img", data.file_name + "_rank_reduced")
    os.makedirs(output_dir, exist_ok = True) # Create the directory if it doesn't exist without raising an error if it exists

    for i in range(len(data.rank)):
        output_path = output_dir + "/" + data.full_name_rank[i]

        img_rank[i].save(output_path) # Save the compressed image in the output directory
        print("   " + data.full_name_rank[i] + " saved")
    print()


"Shows the size of the image"
def sizeIMG():
   # Size of the images
    size_original = os.path.getsize(data.src_img + data.full_name)
    size_rank = []
    for r in data.full_name_rank:
        size_rank.append(os.path.getsize((data.src_img + data.file_name + "_rank_reduced") + "/" + r))

    print()
    print("Original size " + size_formating(size_original))
    print("=" * 25)

    n = len(data.rank)
    for i in range(len(data.rank)):
        print("Rank " + str(data.rank[(n -1) -i]) + " reduced size " + size_formating(size_rank[(n -1) -i]))
        print("Compression ratio " + str(round(size_original / size_rank[(n -1) -i], 2)))
        print("-" * 25)


"Shows the images with matplotlib"
def displayIMG(img):
    n = len(img)  # Number of images to plot

    rank_ = data.rank.copy()
    rank_.append(0) # Insert the original image rank at the end of the list

    # Calculate the number of rows and columns for a square layout
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Set up the subplot grid
    _, axes = plt.subplots(rows, cols)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each image
    for i in range(n):
        axes[i].set_title("Rank " + str(rank_[(n -1) -i]))
        axes[i].imshow(img[(n -1) -i])
        axes[i].axis("off")  # Hide axes for a cleaner look

    # Hide any remaining axes if n is not a perfect square
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()