import os
import numpy as np
import json
from blender.cam_pose_utils.cam_reader import readColmapSceneInfo

if __name__ == "__main__":
    images_folder = "C:\\Users\\10527\\Desktop\\Research of 2DGS\\LiVGaussianMeshing\\data\\jinguilou_post\\images_3.0"
    # get predicted depths
    parent = os.path.dirname(images_folder)
    estimated_depths_folder = os.path.join(parent, "estimated_depths")
    values=readColmapSceneInfo(parent)
    scales = os.path.join(parent, "estimated_depth_scales.json")
    f = open(scales, "r", encoding="utf-8")
    scales = json.load(f)
    for value in values:
        image_name = value.image_name+".jpg"
        predicted_depth = np.load(os.path.join(estimated_depths_folder,image_name+".npy"))
        scale = scales[image_name]["scale"]
        offset = scales[image_name]["offset"]

    # get Lidar depths