import os

# Rank reduction
rank = [5, 10, 25, 50, 100]

# Absolute path to the image
file_name = 'polytech'
extension = '.jpeg'
src_img = './img/'


full_name_rank = []
for r in rank:
    full_name_rank.append(src_img + file_name + '_rank_' + str(r) + extension)

full_name = src_img + file_name + extension
absolute_path = os.path.abspath(full_name)