import concurrent.futures
import json
import math
import os.path
import threading
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track
import random
from typing import Literal, Tuple, Optional, Any
from PIL import Image
import numpy as np
import cv2
import torch.utils.data
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from internal.cameras.cameras import CameraType, Camera
from internal.dataparsers import DataParserConfig, ImageSet
from internal.utils.graphics_utils import store_ply, BasicPointCloud

from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_set: ImageSet,#传入图像集
            undistort_image: bool = True,
            camera_device: torch.device = None,
            image_device: torch.device = None,
            image_uint8: bool = False,
    ) -> None:
        super().__init__()
        self.image_set = image_set
        self.undistort_image = undistort_image

        if camera_device is None:
            camera_device = torch.device("cpu")
        if image_device is None:
            image_device = torch.device("cpu")
            #默认都在cpu上计算
        self.camera_device = camera_device
        self.image_device = image_device
        self.image_uint8 = image_uint8

        self.image_cameras: list[Camera] = [i.to_device(camera_device) for i in image_set.cameras]  # store undistorted camera

    def __len__(self):
        return len(self.image_set)

    def get_image(self, index) -> Tuple[str, torch.Tensor, Optional[torch.Tensor]]:#获取指定索引的图像
        if self.image_set.image_paths[index] is None:#未传入图像
            return self.image_set.image_names[index], None, None

        # TODO: resize
        pil_image = Image.open(self.image_set.image_paths[index])#用PIL库加载指定图像
        numpy_image = np.array(pil_image, dtype=np.uint8)#将图像转换为np数组

        # undistort image
        if self.undistort_image is True:#去除畸变
            assert self.image_uint8 == False
            # TODO: validate this undistortion implementation
            camera = self.image_set.cameras[index]  # get original camera
            distortion = camera.distortion_params
            if distortion is not None and torch.any(distortion != 0.):

                # TODO: support fisheye camera model
                assert camera.camera_type == CameraType.PERSPECTIVE#只支持透视相机

                # build intrinsics matrix
                intrinsics_matrix = np.eye(3)
                intrinsics_matrix[0, 0] = float(camera.fx)  # fx
                intrinsics_matrix[1, 1] = float(camera.fy)  # fy
                intrinsics_matrix[0, 2] = float(camera.cx)  # cx
                intrinsics_matrix[1, 2] = float(camera.cy)  # cy
                # calculate new intrinsics matrix, without black border
                image_shape = (int(camera.width), int(camera.height))
                distortion = distortion.numpy()#畸变参数
                new_intrinsics_matrix, _ = cv2.getOptimalNewCameraMatrix(#去畸变矫正内参
                    intrinsics_matrix,
                    distortion,
                    image_shape,
                    0,
                    image_shape,
                )
                # undistort image
                #根据新的内参得到矫正后的图像
                undistorted_image = cv2.undistort(numpy_image, intrinsics_matrix, distortion, None, new_intrinsics_matrix)
                # update variables
                numpy_image = undistorted_image
                # update image camera
                self.image_cameras[index].camera_type = torch.tensor(CameraType.PERSPECTIVE)
                self.image_cameras[index].fx = torch.tensor(new_intrinsics_matrix[0, 0], dtype=torch.float)
                self.image_cameras[index].fy = torch.tensor(new_intrinsics_matrix[1, 1], dtype=torch.float)
                self.image_cameras[index].cx = torch.tensor(new_intrinsics_matrix[0, 2], dtype=torch.float)
                self.image_cameras[index].cy = torch.tensor(new_intrinsics_matrix[1, 2], dtype=torch.float)
                self.image_cameras[index].distortion_params = torch.zeros((4,), dtype=torch.float)

                if "PREVIEW_UNDISTORTED_IMAGE" in os.environ:#环境变量
                    undistorted_pil_image = Image.fromarray(undistorted_image)
                    image_save_path = os.path.join(os.environ["PREVIEW_UNDISTORTED_IMAGE"], self.image_set.image_names[index])
                    #保存矫正后的图像
                    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                    undistorted_pil_image.save(image_save_path, quality=100)

        if self.image_uint8:
            image = torch.from_numpy(numpy_image)#图像张量
            assert image.dtype == torch.uint8#保证图像以8位无符号整数存储
            assert image.shape[2] == 3#保证三通道
        else:
            image = torch.from_numpy(numpy_image.astype(np.float64) / 255.0)#归一化

            # remove alpha channel将第四个通道融合进前三个RGB通道
            if image.shape[2] == 4:
                # TODO: sync background color with model.background_color
                background_color = torch.tensor([0., 0., 0.])
                image = image[:, :, :3] * image[:, :, 3:4] + background_color * (1 - image[:, :, 3:4])
                #image[:, :, 3:4]是透明度的通道
            image = image.to(torch.float)

        mask = None
        if self.image_set.mask_paths[index] is not None:
            pil_image = Image.open(self.image_set.mask_paths[index])#mask图像
            mask = torch.from_numpy(np.array(pil_image))
            
            # mask must be single channel
            assert len(mask.shape) == 2, "the mask image must be single channel"

            # the shape of the mask must match to the image
            assert mask.shape[:2] == image.shape[:2], \
                "the shape of mask {} doesn't match to the image {}".format(mask.shape[:2], image.shape[:2])
            
            mask = (mask == 0).unsqueeze(-1).expand(*image.shape)  # True is the masked pixels
            #扩展mask图像使其大小与彩色图像一致

            mask = mask.permute(2, 0, 1).to(self.image_device)  # [channel, height, width]  CHW格式

        image = image.permute(2, 0, 1).to(self.image_device)  # [channel, height, width]  转换彩色图像的格式

        return self.image_set.image_names[index], image, mask#得到索引号，图像张量和mask张量

    def get_extra_data(self, index):
        return self.image_set.extra_data_processor(self.image_set.extra_data[index])

    def __getitem__(self, index) -> Tuple[Camera, Tuple, Any]:
        return self.image_cameras[index], self.get_image(index), self.get_extra_data(index)


class CacheDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            max_cache_num: int,
            shuffle: bool,#是否打乱顺序
            seed: int = -1,
            distributed: bool = False,
            world_size: int = -1,
            global_rank: int = -1,
            async_caching: bool = False,
            **kwargs,#额外不定数量的参数
    ):
        assert kwargs.get("batch_size", 1) == 1, "only batch_size=1 is supported"

        self.dataset = dataset

        super().__init__(dataset=dataset, **kwargs)

        self.shuffle = shuffle

        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        #此处已经调整了内存分配策略

        self.max_cache_num = max_cache_num=1

        # image indices to use
        self.indices = list(range(len(self.dataset)))#保存所有的索引

        if distributed is True and self.max_cache_num != 0:#分布式处理
            assert world_size > 0
            assert global_rank >= 0

            #为每个进程划分使用的图像数量
            image_num_to_use = math.ceil(len(self.indices) / world_size)
            start = global_rank * image_num_to_use
            end = start + image_num_to_use
            indices = self.indices[start:end]
            indices += self.indices[:image_num_to_use - len(indices)]
            self.indices = indices

            print("#{} distributed indices (total: {}): {}".format(os.getpid(), len(self.indices), self.indices))

        # cache all images if max_cache_num > len(dataset)
        if self.max_cache_num >= len(self.indices):
            self.max_cache_num = -1

        self.num_workers = kwargs.get("num_workers", 0)

        if self.max_cache_num < 0:
            # cache all data
            print("cache all images")
            self.cached = self._cache_data(self.indices)

        # use dedicated random number generator foreach dataloader
        if self.shuffle is True:
            assert seed >= 0, "seed must be provided when shuffle=True"
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            print("#{} dataloader seed to {}".format(os.getpid(), seed))

        #异步缓存
        self.async_caching = async_caching and self.max_cache_num > 0
        self.cache_output_queue = None
        self.cache_thread = None
        self.stop_caching = False
        if self.async_caching:
            self.cache_output_queue = queue.Queue(maxsize=1)
            self.cache_thread = threading.Thread(target=self._async_cache)
            self.cache_thread.start()

    def _async_cache(self):
        # TODO: GC will freeze program a while
        while not self.stop_caching:#当所有数据都加载过一轮后，若未达到指定步数就继续重新加载数据
            if self.shuffle is True:
                indices = torch.randperm(len(self.indices), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            else:
                indices = self.indices.copy()

            not_cached = indices.copy()

            while not_cached and not self.stop_caching:
                # select self.max_cache_num images
                to_cache = not_cached[:self.max_cache_num]
                del not_cached[:self.max_cache_num]#将已写入缓存的图像移除

                self.cache_output_queue.put(None)  # simulate a queue with zero size
                self.cache_output_queue.put(self._cache_data(to_cache, pbar_leave=False))
                #将缓存的数据放入输出队列中

    def _cache_data(self, 
                    indices: list, #需要缓存的数据索引
                    pbar_leave: bool = True#是否在进度结束时保留进度条
                    ):
        cached = []
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:#线程池执行器
                for i in tqdm(
                        e.map(self.dataset.__getitem__, indices),#并行对每个索引对应的数据进行读取
                        #每次只对缓存的数据进行读取并返回

                        total=len(indices),#数据中总共的数量
                        desc="#{} caching images (1st: {})".format(os.getpid(), indices[0]),
                        leave=pbar_leave,
                ):
                    cached.append(i)
        else:
            for i in tqdm(indices, desc="#{} loading images (1st: {})".format(os.getpid(), indices[0]), leave=pbar_leave):
                cached.append(self.dataset.__getitem__(i))

        return cached#返回读取的数据

    def __len__(self) -> int:
        return len(self.indices)#需要读取的所有数据总量

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __iter__(self):
        # TODO: support batching
        if self.max_cache_num < 0:#缓存所有数据
            if self.shuffle is True:
                indices = torch.randperm(len(self.cached), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            else:
                indices = list(range(len(self.cached)))

            for i in indices:
                yield self.cached[i]#返回所有数据的迭代
        else:
            if self.shuffle is True:
                indices = torch.randperm(len(self.indices), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            else:
                indices = self.indices.copy()

            # print("#{} self.max_cache_num={}, indices: {}".format(os.getpid(), self.max_cache_num, indices))

            if self.max_cache_num == 0:
                # no cache
                for i in indices:
                    yield self.__getitem__(i)
            else:
                # cache
                # the list contains the data have not been cached
                not_cached = indices.copy()

                if self.async_caching:
                    while True:
                        cached = self.cache_output_queue.get()  # setting to None allows GC
                        assert cached is None
                        cached = self.cache_output_queue.get()
                        for i in cached:
                            yield i

                #未选择异步缓存            
                else:
                    while not_cached:
                        # select self.max_cache_num images
                        to_cache = not_cached[:self.max_cache_num]
                        del not_cached[:self.max_cache_num]

                        # cache
                        try:
                            del cached
                        except:
                            pass
                        cached = self._cache_data(to_cache, pbar_leave=False)

                        for i in cached:
                            yield i


class DataModule(LightningDataModule):
    def __init__(
            self,
            path: str,
            parser: DataParserConfig = None,
            distributed: bool = False,
            undistort_image: bool = False,
            val_on_train: bool = False,
            image_scale_factor: float = 1.,  # TODO
            train_max_num_images_to_cache: int = 1,
            val_max_num_images_to_cache: int = -1,
            test_max_num_images_to_cache: int = -1,
            num_workers: int = 2,
            add_background_sphere: bool = False,
            background_sphere_center: Literal["points", "cameras"] = "points",
            background_sphere_distance: float = 2.2,
            background_sphere_points: int = 204_800,
            background_sphere_color: Literal["random", "white"] = "random",
            background_sphere_min_altitude: float = -math.inf,
            camera_on_cpu: bool = False,
            image_on_cpu: bool = True,
            image_uint8: bool = False,
            async_caching: bool = False,
    ) -> None:
        r"""Load dataset

            Args:
                path: the path to the dataset

                type: the dataset type
        """

        super().__init__()

        assert image_scale_factor == 1., f"specifying 'image_scale_factor' has not been implemented yet"

        if parser is None:
            parser = self.detect_dataset_type(path)#根据不同的路径文件类型选择合适的解析器
            print(f"Detected dataset type: {parser.__class__.__name__}")

        self.save_hyperparameters()#保存超参数

        self.camera_device = torch.device("cpu")
        self.image_device = torch.device("cpu")

    def set_device(self, device):
        if self.hparams["camera_on_cpu"] is False:
            self.camera_device = device
        if self.hparams["image_on_cpu"] is False:
            self.image_device = device

    @staticmethod
    def detect_dataset_type(path):
        if os.path.isdir(os.path.join(path, "sparse")) is True:
            from internal.dataparsers.colmap_dataparser import Colmap
            return Colmap()
        elif os.path.exists(os.path.join(path, "transforms_train.json")):
            from internal.dataparsers.blender_dataparser import Blender
            return Blender()
        elif os.path.exists(os.path.join(path, "intrinsics.txt")) and os.path.exists(os.path.join(path, "bbox.txt")):
            from internal.dataparsers.nsvf_dataparser import NSVF
            return NSVF()
        elif os.path.exists(os.path.join(path, "dataset.json")):
            from internal.dataparsers.nerfies_dataparser import Nerfies
            return Nerfies()
        else:
            raise ValueError("Can not detect dataset type, please specify via '--data.parser'")

    def setup(self, stage: str) -> None:
        super().setup(stage)

        output_path = self.trainer.lightning_module.hparams["output_path"]#解析输出路径

        # store global rank, will be used as the seed of the CacheDataLoader
        self.global_rank = self.trainer.global_rank

        dataparser = self.hparams["parser"].instantiate(path=self.hparams["path"], output_path=output_path, global_rank=self.global_rank)
        #instantiate函数返回一个dataparser对象，参数由数据路径、输出路径、编号组成

        # load dataset
        self.dataparser_outputs = dataparser.get_outputs()#将超参数转换为字典的形式

        self.prune_extent = self.dataparser_outputs.camera_extent

        # add background sphere: https://github.com/graphdeco-inria/gaussian-splatting/issues/300#issuecomment-1756073909
        if self.hparams["add_background_sphere"] is True:
            # find the scene center and size
            if self.hparams["background_sphere_center"] == "points":
                scene_center = self.dataparser_outputs.point_cloud.xyz.mean(axis=0)
                scene_radius = np.percentile(np.linalg.norm(self.dataparser_outputs.point_cloud.xyz - scene_center, axis=-1), 99.9).item()
                #计算所有点云到场景中心的距离最大值
            else:#以摄像机为场景中心
                scene_center = self.dataparser_outputs.train_set.cameras.camera_center.mean(dim=0)#计算所有摄像机中心的平均位置，作为场景中心
                scene_radius_from_cameras = torch.norm(self.dataparser_outputs.train_set.cameras.camera_center - scene_center, dim=-1).max().item()
                #所有摄像机到场景中心的最大值

                scene_center = scene_center.numpy()
                radius_from_points = np.percentile(np.linalg.norm(self.dataparser_outputs.point_cloud.xyz - scene_center, axis=-1), 99.9).item()
                #所有点云到场景中心的距离最大值

                scene_radius = max(scene_radius_from_cameras, radius_from_points)#取最大值
                scene_radius = scene_radius

            # build unit sphere points单位球面上均匀采样生成n个点
            n_points = self.hparams["background_sphere_points"]#以点云构建球状背景
            samples = np.arange(n_points)
            y = 1 - (samples / float(n_points - 1)) * 2  # y goes from 1 to -1
            #从1至-1
            
            radius = np.sqrt(1 - y * y)  # radius at y
            #y处圆的半径

            phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

            theta = phi * samples  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            #每个radius对应一个samples，每个theta也对应一个samples

            unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)#螺旋形采样
            # build background sphere
            background_sphere_point_xyz = (unit_sphere_points * scene_radius * self.hparams["background_sphere_distance"]) + scene_center

            # simply delete those under min altitude
            # TODO: custom up direction
            background_sphere_point_xyz = background_sphere_point_xyz[background_sphere_point_xyz[:, -1] >= self.hparams["background_sphere_min_altitude"]]

            if self.hparams["background_sphere_color"] == "random":
                background_sphere_point_rgb = np.asarray(np.random.random(background_sphere_point_xyz.shape) * 255, dtype=np.uint8)
            else:
                background_sphere_point_rgb = np.ones(background_sphere_point_xyz.shape, dtype=np.uint8) * 255
            # add background sphere to scene
            self.dataparser_outputs.point_cloud.xyz = np.concatenate([self.dataparser_outputs.point_cloud.xyz, background_sphere_point_xyz], axis=0)
            self.dataparser_outputs.point_cloud.rgb = np.concatenate([self.dataparser_outputs.point_cloud.rgb, background_sphere_point_rgb], axis=0)
            # increase prune extent
            # TODO: resize scene_extent without changing lr
            self.prune_extent = scene_radius * self.hparams["background_sphere_distance"] * 1.0001

            print("added {} background sphere points, scene_center={}, scene_radius={}, rescale prune extent from {} to {}".format(background_sphere_point_xyz.shape, scene_center.tolist(), scene_radius, self.dataparser_outputs.camera_extent, self.prune_extent))

        # convert point cloud
        self.point_cloud = self.dataparser_outputs.point_cloud

        # write some files that SIBR_viewer required
        if self.global_rank == 0 and stage == "fit":

            # write appearance group id
            if self.dataparser_outputs.appearance_group_ids is not None:
                torch.save(
                    self.dataparser_outputs.appearance_group_ids,
                    os.path.join(output_path, "appearance_group_ids.pth"),
                )
                with open(os.path.join(output_path, "appearance_group_ids.json"), "w") as f:
                    json.dump(self.dataparser_outputs.appearance_group_ids, f, indent=4, ensure_ascii=False)

            # write cameras.json
            camera_to_world = torch.linalg.inv(
                torch.transpose(self.dataparser_outputs.train_set.cameras.world_to_camera, #world_to_camera存储的是转置，因此要先转置回来
                                1, 
                                2)#转置第一维和第二维
            ).numpy()
            cameras = []
            for idx, image in enumerate(self.dataparser_outputs.train_set):
                image_name, _, _, camera, _ = image
                cameras.append({#添加摄像机信息
                    'id': idx,
                    'img_name': image_name,
                    'width': int(camera.width),
                    'height': int(camera.height),
                    'position': camera_to_world[idx, :3, 3].tolist(),
                    'rotation': [x.tolist() for x in camera_to_world[idx, :3, :3]],
                    'fy': float(camera.fy),
                    'fx': float(camera.fx),
                    'cx': camera.cx.item(),
                    'cy': camera.cy.item(),
                    'time': camera.time.item() if camera.time is not None else None,
                    'appearance_id': camera.appearance_id.item() if camera.appearance_id is not None else None,
                    'normalized_appearance_id': camera.normalized_appearance_id.item() if camera.normalized_appearance_id is not None else None,
                })
            with open(os.path.join(output_path, "cameras.json"), "w") as f:
                json.dump(cameras, f, indent=4, ensure_ascii=False)

            # save input point cloud to ply file
            store_ply(#将点写入点云文件
                os.path.join(output_path, "input.ply"),
                xyz=self.dataparser_outputs.point_cloud.xyz,
                rgb=self.dataparser_outputs.point_cloud.rgb,
            )

            # write cfg_args
            try:
                with open(os.path.join(output_path, "cfg_args"), "w") as f:
                    f.write("Namespace(sh_degree={}, white_background={}, source_path='{}', images='images', eval=True, resolution=1, data_device='cpu')".format(
                        self.trainer.lightning_module.hparams["gaussian"].sh_degree,
                        True if torch.all(self.trainer.lightning_module.background_color == 1.) else False,
                        self.hparams["path"],
                    ))
            except:
                pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return CacheDataLoader(
            Dataset(
                self.dataparser_outputs.train_set,
                undistort_image=self.hparams["undistort_image"],
                camera_device=self.camera_device,
                image_device=self.image_device,
                image_uint8=self.hparams["image_uint8"],
            ),
            max_cache_num=self.hparams["train_max_num_images_to_cache"],
            shuffle=True,
            seed=torch.initial_seed() + self.global_rank,  # seed with global rank
            num_workers=self.hparams["num_workers"],
            distributed=self.hparams["distributed"],
            world_size=self.trainer.world_size,
            global_rank=self.trainer.global_rank,
            async_caching=self.hparams["async_caching"],
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.hparams["val_on_train"] is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.test_set
        return CacheDataLoader(
            Dataset(
                image_set,
                undistort_image=self.hparams["undistort_image"],
                camera_device=self.camera_device,
                image_device=self.image_device,
                image_uint8=self.hparams["image_uint8"],
            ),
            max_cache_num=self.hparams["test_max_num_images_to_cache"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.hparams["val_on_train"] is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.val_set
        return CacheDataLoader(
            Dataset(
                image_set,
                undistort_image=self.hparams["undistort_image"],
                camera_device=self.camera_device,
                image_device=self.image_device,
                image_uint8=self.hparams["image_uint8"],
            ),
            max_cache_num=self.hparams["val_max_num_images_to_cache"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if batch[1][1].dtype != torch.uint8:
            return batch

        camera, image_info, extra_data = batch
        image_name, gt_image, masked_pixels = image_info

        gt_image = gt_image.to(camera.R.dtype) / 255.

        return camera, (image_name, gt_image, masked_pixels), extra_data
