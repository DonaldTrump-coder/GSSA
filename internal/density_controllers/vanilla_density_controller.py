from typing import Tuple, Optional, Union, List, Dict
from dataclasses import dataclass
import torch
from torch import nn
from lightning import LightningModule

from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.general_utils import build_rotation
from .density_controller import DensityController, DensityControllerImpl, Utils


@dataclass
class VanillaDensityController(DensityController):
    percent_dense: float = 0.007

    densification_interval: int = 150

    opacity_reset_interval: int = 33000

    densify_from_iter: int = 2000

    densify_until_iter: int = 18_000

    densify_grad_threshold: float = 0.0001

    cull_opacity_threshold: float = 0.006
    """threshold of opacity for culling gaussians."""

    camera_extent_factor: float = 1.

    scene_extent_override: float = -1.

    absgrad: bool = False

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return VanillaDensityControllerImpl(self)


class VanillaDensityControllerImpl(DensityControllerImpl):
    #不根据梯度更新参数，只依据梯度进行密度控制
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        if stage == "fit":
            self.cameras_extent = pl_module.trainer.datamodule.dataparser_outputs.camera_extent * self.config.camera_extent_factor
            self.prune_extent = pl_module.trainer.datamodule.prune_extent * self.config.camera_extent_factor

            if self.config.scene_extent_override > 0:
                self.cameras_extent = self.config.scene_extent_override
                self.prune_extent = self.config.scene_extent_override
                print(f"Override scene extent with {self.config.scene_extent_override}")

            self._init_state(pl_module.gaussian_model.n_gaussians, pl_module.device)

    def _init_state(self, n_gaussians: int, device):
        max_radii2D = torch.zeros((n_gaussians), device=device)
        xyz_gradient_accum = torch.zeros((n_gaussians, 1), device=device)
        denom = torch.zeros((n_gaussians, 1), device=device)

        self.register_buffer("max_radii2D", max_radii2D, persistent=True)
        self.register_buffer("xyz_gradient_accum", xyz_gradient_accum, persistent=True)
        self.register_buffer("denom", denom, persistent=True)#参数不参与反向传播

    def before_backward(self, 
                        outputs: dict, #前向传播的输出
                        batch, 
                        gaussian_model: VanillaGaussianModel, 
                        optimizers: List, 
                        global_step: int, #当前步数
                        pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return#当前训练步数已经超过了指定的最大步数

        outputs["viewspace_points"].retain_grad()#只保留看得到的点的梯度

    def after_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        #未超过最大步数时进行高斯的加密和修剪
        with torch.no_grad():#不计算梯度
            self.update_states(outputs)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
            #密度控制步骤
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                )

            """
            if global_step % self.config.opacity_reset_interval == 0 or \
                    (
                            torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                            #背景是白色
                    ):
                self._reset_opacities(gaussian_model, optimizers)#重置透明度
            """

    def update_states(self, outputs):
        viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update states
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )#某个可见高斯点的最大半径存储

        xys_grad = viewspace_point_tensor.grad#计算反向梯度张量
        if self.config.absgrad is True:
            xys_grad = viewspace_point_tensor.absgrad#计算绝对梯度张量
        self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)

    def _add_densification_stats(self, #记录每个点的梯度相关状态
                                 grad, #表示梯度值的张量
                                 update_filter, #更新高斯的索引
                                 scale: Union[float, int, None]#对梯度进行缩放
                                 ):
        scaled_grad = grad[update_filter, :2]
        if scale is not None:
            scaled_grad = scaled_grad * scale#进行梯度缩放
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)
        #计算每个点梯度的范数

        self.xyz_gradient_accum[update_filter] += grad_norm#存入每个点的梯度范数累计值
        self.denom[update_filter] += 1#记录每个点更新的次数

    def _densify_and_prune(self, max_screen_size, gaussian_model: VanillaGaussianModel, optimizers: List):
        min_opacity = self.config.cull_opacity_threshold#低于此透明度的高斯会被修剪掉
        prune_extent = self.prune_extent#修剪范围

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom#计算平均每个高斯的梯度
        grads[grads.isnan()] = 0.0

        # densify
        self._densify_and_clone(grads, gaussian_model, optimizers)
        self._densify_and_split(grads, gaussian_model, optimizers)

        # prune
        prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
        if max_screen_size:#修剪空间过大的高斯
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # Exclude big Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():#键值对，key是属性名，value是所有点的属性值
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

    def _split_means_and_scales(self, gaussian_model, selected_pts_mask, N):
        scales = gaussian_model.get_scales()
        device = scales.device

        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussian_model.get_property("rotations")[selected_pts_mask]).repeat(N, 1, 1)
        # Split means and scales, they are a little bit different
        new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussian_model.get_means()[selected_pts_mask].repeat(N, 1)
        new_scales = gaussian_model.scale_inverse_activation(scales[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        new_properties = {
            "means": new_means,
            "scales": new_scales,
        }

        return new_properties

    def _split_properties(self, gaussian_model, selected_pts_mask, N: int):
        new_properties = self._split_means_and_scales(gaussian_model, selected_pts_mask, N)

        # Split other properties
        for key, value in gaussian_model.properties.items():
            if key in new_properties:
                continue
            new_properties[key] = value[selected_pts_mask].repeat(N, *[1 for _ in range(value[selected_pts_mask].dim() - 1)])

        return new_properties

    def _densify_and_split(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List, N: int = 2):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # Exclude small Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(
                scales,
                dim=1,
            ).values > percent_dense * scene_extent,
        )

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * selected_pts_mask.sum(),
                device=device,
                dtype=torch.bool,
            ),
        ))
        self._prune_points(prune_filter, gaussian_model, optimizers)

    def _densification_postfix(self, new_properties: Dict, gaussian_model, optimizers):
        new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # re-init states
        self._init_state(gaussian_model.n_gaussians, gaussian_model.get_property("means").device)

    def _prune_points(self, mask, gaussian_model: VanillaGaussianModel, optimizers: List):
        """
        Args:
            mask: `True` indicating the Gaussians to be pruned
            gaussian_model
            optimizers
        """
        valid_points_mask = ~mask  # `True` to keep
        new_parameters = Utils.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # prune states
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _reset_opacities(self, gaussian_model: VanillaGaussianModel, optimizers: List):
        opacities_new = gaussian_model.opacity_inverse_activation(torch.min(
            gaussian_model.get_opacities(),
            torch.ones_like(gaussian_model.get_opacities()) * 0.01,
        ))
        new_parameters = Utils.replace_tensors_to_properties(tensors={
            "opacities": opacities_new,
        }, optimizers=optimizers)
        gaussian_model.update_properties(new_parameters)

    def on_load_checkpoint(self, module, checkpoint):
        self._init_state(checkpoint["state_dict"]["density_controller.max_radii2D"].shape[0], module.device)

    def after_density_changed(self, gaussian_model, optimizers: List, pl_module: LightningModule) -> None:
        self._init_state(gaussian_model.n_gaussians, pl_module.device)
