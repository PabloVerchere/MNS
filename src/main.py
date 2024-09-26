from PIL import Image
import time

import fct
import data



# Start time
start_time = time.time()

# Original image loading
img = Image.open(data.absolute_path)
print('Image ' + data.file_name + ' loaded')
print()


# SVD compression
img_rank = fct.svd(img, data.rank)


# Save the compressed images
print('Saving compressed images...')
for i in range(len(data.rank)):
    img_rank[i].save(data.full_name_rank[i])
    print('   ' +  data.file_name + '_rank_' + str(data.rank[i]) + data.extension)
print()


# End time
end_time = time.time()

print('=' * 25)
print('Elapsed time: ' + str(round(end_time - start_time, 2)) + ' seconds')
print('=' * 25)


fct.sizeIMG()

img_rank.append(img) # Insert the original image at the end of the list
fct.displayIMG(img_rank)