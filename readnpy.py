import numpy as np

datagt=np.load("data/matrix_city/aerial/train/block_all/sampled_depths/3453.png.npy")
#datapd=np.load("data/matrix_city/aerial/train/block_all/estimated_depths/3363.png.npy")
datapd=np.load("data/matrix_city/aerial/train/block_all/predicted_depths/3453.png.npy")
scale=1
offset=0
print(1/(datagt[40,12]*scale+offset))
print(datapd[40,12])