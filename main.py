import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT_DIR, 'images')

cat = np.array(cv.imread(os.path.join(IMAGE_DIR, 'cat.jpg'))) 
cat = np.array(cv.cvtColor(cat, cv.COLOR_BGR2RGB))

# Rank approximation of the matrix
ranks = [5, 10, 20, 30, 50]

num_rows = 3
num_cols = 2
figure, axis = plt.subplots(3, 2, figsize=(15, 15))

for i, r in enumerate(ranks):
    m = []

    # Calculating U, S, V_T matrix for each color channel
    for c in range(cat.shape[-1]):
        U, S, V_T = np.linalg.svd(cat[:, :, c] / 255., full_matrices=False)
        S = np.diag(S)

        # Reducing the number of dimensions of each decomposition matrix
        m.append(U[:, :r] @ S[:r, :r] @ V_T[:r, :])
    
    row = i // num_cols
    col = i % num_cols

    # Stacking the color channel matrices together back into a 3d tensor 
    m = np.stack(m, axis=-1)
    axis[row, col].set_title(f'r = {r}')
    axis[row, col].imshow(m)

axis[-1, -1].set_title('Original Image')
axis[-1, -1].imshow(cat)
plt.show()

