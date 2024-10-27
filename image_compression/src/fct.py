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
    S = np.diag(S)  # Convert S (a vector) into a diagonal matrix of dim m x m    

    return U, S, V


"Reconstructs the matrix using the SVD components with rank r"
def reconstruct_matrix(U, S, V, r):
    M = U[:, :r] @ S[:r, :r] @ V[:r, :]  # Use rank-reduced components for approximation

    return M


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
    return img_rank, [Sr, Sg, Sb]


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


def clean_file(png = False):
    print("Cleaning compressed images...")

    img_dir = os.path.join("image_compression/img")
    print(img_dir)

    for filename in os.listdir(img_dir):
        if filename.endswith("_rank_reduced"):
            shutil.rmtree(os.path.join(img_dir, filename)) # Remove the directory and its content
        
        if png:
            if filename.endswith("_compressed_images.png") or filename.endswith("_energy.png"):
                os.remove(os.path.join(img_dir, filename)) # Remove the file


def saveIMG(img_rank):
    print("Saving compressed images...")

    # Create the directory to save the compressed images
    output_dir = os.path.join("image_compression", "img", data.file_name + "_rank_reduced")
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
    for i in range(n):
        print("Rank " + str(data.rank[(n -1) -i]) + " reduced size " + size_formating(size_rank[(n -1) -i]))
        print("Compression ratio " + str(round(size_original / size_rank[(n -1) -i], 2)))
        print("-" * 25)
    print()


"Shows the images rank reduced"
def displayIMG(original, img):
    n = len(img)  # Number of images to plot

    rank_ = data.rank.copy()
    rank_.append(min(original.size)) # Insert the original image rank at the end of the list

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


    plt.suptitle(data.file_name)
    plt.savefig("image_compression/img/" + data.file_name + "_compressed_images.png")

    plt.tight_layout()
    plt.show(block = False)


"Verify if the rank is inf to the image size"
def verify_rank(img):
    width, height = img.size
    min_dim = min(width, height)

    nb = 0
    for r in data.rank:
        if r < min_dim:
            nb += 1
    
    data.rank = data.rank[:nb] # Update the rank list with the valid ranks
    data.full_name_rank = data.full_name_rank[:nb] # Update the full name rank list with the valid ranks


def computeEnergy(singular_values):
    return np.cumsum(singular_values) / np.sum(singular_values)  # Compute the cumulative sum of the singular values


"Display the energy depending on the singular values"
def displayEnergy(singular_values):
    # Compute the cumulative energy for each channel
    energy = computeEnergy(singular_values)

    x = np.arange(1, len(energy) + 1)  # Create an array of the same length as the singular values

    # Show the energy for rank reduction
    print("Energy (singular values)")
    print("=" * 25)

    n = len(data.rank)
    for i in range(n):
        print("Rank " + str(data.rank[(n -1) -i]) + " energy conserved " + str(round(energy[data.rank[(n -1) -i]], 2)))
        print("-" * 25)
    print()
    
    # Create subplots
    _, axs = plt.subplots(1, 2)


    axs.flat[0].plot(x, singular_values, marker='o', linestyle = 'None', color='r')
    axs.flat[0].set_xlabel("Number of singular values")
    axs.flat[0].set_ylabel("Singular values")
    axs.flat[0].set_yscale('log') # put the y axis in log scale

    x_lim = axs.flat[0].get_xlim()
    y_lim = axs.flat[0].get_ylim()

    for r in data.rank:
        # Minmax normalization to get the position of the line depending on the limits of the plot (on xmax and ymax)
        axs.flat[0].axvline(x = r, ymax = ((math.log(singular_values[r]) - math.log(y_lim[0])) / (math.log(y_lim[1]) - math.log(y_lim[0]))), color = 'k', linestyle = '--', alpha = 0.4)
        axs.flat[0].axhline(y = singular_values[r], xmax = ((r - x_lim[0]) / (x_lim[1] - x_lim[0])), color = 'k', linestyle = '--', alpha = 0.4)


    axs.flat[1].plot(x, energy, marker='o', linestyle = 'None')
    axs.flat[1].set_xlabel("Number of singular values")
    axs.flat[1].set_ylabel("Energy")

    x_lim = axs.flat[1].get_xlim()
    y_lim = axs.flat[1].get_ylim()

    for r in data.rank:
        # Minmax normalization to get the position of the line depending on the limits of the plot (on xmax and ymax)
        axs.flat[1].axvline(x = r, ymax = ((energy[r] - y_lim[0]) / (y_lim[1] - y_lim[0])), color = 'k', linestyle = '--', alpha = 0.4)
        axs.flat[1].axhline(y = energy[r], xmax = ((r - x_lim[0]) / (x_lim[1] - x_lim[0])), color = 'k', linestyle = '--', alpha = 0.4)

    
    plt.suptitle(data.file_name)
    plt.savefig("image_compression/img/" + data.file_name + "_energy.png")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show(block = False)