from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


@generic_io_descriptor(
    observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def over_orientation(env: ManagerBasedEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor: # 判断姿态是否过大，作为终止条件使用
    """Bad orientation.
    
    Note: This function is typically used as a termination condition.
    
    Args:
        env: The environment.
        limit_angle: The limit angle in radians.
        asset_cfg: The RigidObject associated with this observation.

    Returns:
        A boolean tensor indicating whether the orientation is bad.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs().unsqueeze(-1) > limit_angle # 如果当前机器人Z轴夹角与重力方向的夹角过大，则认为姿态过大，返回True，否则返回False


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_height_w( # 返回指定body在世界坐标系下的高度
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The height of bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The height of the bodies of the Articulation. Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2]


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_lin_vel_w_norm( # 返回指定body在世界坐标系下的水平线速度的模长
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The linear velocity of bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The linear velocity of bodies of an Articulation. Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
