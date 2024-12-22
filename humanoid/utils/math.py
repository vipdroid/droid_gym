# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024,Shanghai Droid Robot CO.,LTD. All rights reserved.


import math

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, get_euler_xyz
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def get_scale_shift(range):
    # scale = 2. / (range[1] - range[0])
    scale = 2. / (range[1] - range[0]) if range[1] != range[0] else 1.
    shift = (range[1] + range[0]) / 2.
    return scale, shift


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


def get_rotation_matrix_from_rpy(rpy):
    """
    根据滚转角（roll）、俯仰角（pitch）和偏航角（yaw）计算旋转矩阵。

    参数:
    rpy (torch.Tensor): 包含滚转角、俯仰角和偏航角的张量，形状为 (..., 3)，单位为弧度。

    返回:
    torch.Tensor: 旋转矩阵，形状为 (..., 3, 3)。
    """
    r, p, y = rpy.unbind(-1)  # 分别提取 roll, pitch, yaw

    # 计算绕 X 轴的旋转矩阵
    R_x = torch.stack([
        torch.ones_like(r), torch.zeros_like(r), torch.zeros_like(r),
        torch.zeros_like(r), torch.cos(r), -torch.sin(r),
        torch.zeros_like(r), torch.sin(r), torch.cos(r)
    ], dim=-1).reshape(*r.shape, 3, 3)

    # 计算绕 Y 轴的旋转矩阵
    R_y = torch.stack([
        torch.cos(p), torch.zeros_like(p), torch.sin(p),
        torch.zeros_like(p), torch.ones_like(p), torch.zeros_like(p),
        -torch.sin(p), torch.zeros_like(p), torch.cos(p)
    ], dim=-1).reshape(*p.shape, 3, 3)

    # 计算绕 Z 轴的旋转矩阵
    R_z = torch.stack([
        torch.cos(y), -torch.sin(y), torch.zeros_like(y),
        torch.sin(y), torch.cos(y), torch.zeros_like(y),
        torch.zeros_like(y), torch.zeros_like(y), torch.ones_like(y)
    ], dim=-1).reshape(*y.shape, 3, 3)

    # 组合旋转矩阵：R = R_z * R_y * R_x
    rot = torch.matmul(R_z, torch.matmul(R_y, R_x))
    return rot


def get_gravity_vector(rot):
    """
    根据旋转矩阵计算重力向量。

    参数:
    rot (torch.Tensor): 旋转矩阵，形状为 (..., 3, 3)。

    返回:
    torch.Tensor: 重力向量，形状为 (..., 3)。
    """
    gravity_direction = torch.tensor([0, 0, -1], dtype=rot.dtype, device=rot.device)
    # 扩展 gravity_direction 以匹配 rot 的维度
    gravity_direction = gravity_direction.view(*([1] * (rot.dim() - 1)), 3)
    grav = torch.matmul(rot, gravity_direction.transpose(-1, -2)).squeeze(-1)
    return grav

# @ torch.jit.script
def exp_avg_filter(x, avg, alpha):
    """
    Simple exponential average filter
    """
    avg = alpha*x + (1-alpha)*avg
    return avg


def apply_coupling(q, qd, q_des, qd_des, kp, kd, tau_ff):
    # Create a Jacobian matrix and move it to the same device as input tensors
    J = torch.eye(q.shape[-1]).to(q.device)
    J[4, 3] = 0.2
    J[9, 8] = 0.2

    # Perform transformations using Jacobian
    q = torch.matmul(q, J.T)
    qd = torch.matmul(qd, J.T)
    q_des = torch.matmul(q_des, J.T)
    qd_des = torch.matmul(qd_des, J.T)

    # Inverse of the transpose of Jacobian
    J_inv_T = torch.inverse(J.T)

    # Compute feed-forward torques
    tau_ff = torch.matmul(J_inv_T, tau_ff.T).T

    # Compute kp and kd terms
    kp = torch.diagonal(
        torch.matmul(
            torch.matmul(J_inv_T, torch.diag_embed(kp, dim1=-2, dim2=-1)),
            J_inv_T.T
        ),
        dim1=-2, dim2=-1
    )

    kd = torch.diagonal(
        torch.matmul(
            torch.matmul(J_inv_T, torch.diag_embed(kd, dim1=-2, dim2=-1)),
            J_inv_T.T
        ),
        dim1=-2, dim2=-1
    )

    # Compute torques
    torques = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
    torques = torch.matmul(torques, J)

    return torques
