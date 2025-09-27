import matplotlib.cm
import numpy as np
from PIL import Image
import matplotlib
colormap=matplotlib.cm.plasma

scale=0.005949547978399041
offset=0.009840343493128387
depth_anything_file="data/jinguilou/predicted_depths/0004.png.npy"
training_depth_file="data/jinguilou/estimated_depths/00004_DJI_20241114155632_0006_Zenmuse-L1-mission.JPG.npy"

depth_anything=np.load(depth_anything_file)
depth_anything=1/(depth_anything*scale+offset)
training_depth=np.load(training_depth_file)
print(depth_anything)
print(training_depth)

base=np.max([np.max(depth_anything),np.max(training_depth)])

depth_anything/=base
#depth_anything = colormap(depth_anything)[:, :, :3]
depth_anything = (depth_anything * 255).astype(np.uint8)


training_depth/=base
#training_depth = colormap(training_depth)[:, :, :3]
training_depth = (training_depth * 255).astype(np.uint8)

depth_anything_img=Image.fromarray(depth_anything)
depth_anything_img.show()
training_depth_img=Image.fromarray(training_depth)
training_depth_img.show()