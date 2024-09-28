from PIL import Image
import time

import fct
import data
import os


# Ask for the complete name of the image
image_name = input("Please enter the complete name of the image: ")

# Extract the file name and extension
data.file_name, data.extension = os.path.splitext(image_name)

# Update the full name and absolute path
fct.update_full_name()

# Clean the img folder
fct.clean_file(1)


# Start time
start_time = time.time()

# Original image loading
img = Image.open(data.absolute_path)
print("Image " + data.file_name + " loaded")
print()

# SVD compression
img_rank = fct.svd(img, data.rank)

# Save the compressed images
fct.saveIMG(img_rank)

# End time
end_time = time.time()


print("=" * 25)
print("Elapsed time: " + str(round(end_time - start_time, 2)) + " seconds")
print("=" * 25)

fct.sizeIMG()

img_rank.append(img) # Insert the original image at the end of the list
fct.displayIMG(img_rank)