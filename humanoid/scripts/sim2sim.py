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
from pynput import keyboard
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import x02Cfg
import torch


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):

    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y])


def get_obs(data):

    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('gyro').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(default_pos, target_q, q, kp, target_dq, dq, kd):

    return (target_q - q + default_pos) * kp + (target_dq - dq) * kd

vx, vy, dyaw = 0.0, 0.0, 0.0


def on_press(key):
    global vx, vy, dyaw
    try:
        if key.char == '2':  # 向前
            vx += 0.1
        elif key.char == '3':  # 向后
            vx -= 0.1
        elif key.char == '4':  # 向左
            vy += 0.1
        elif key.char == '5':  # 向右
            vy -= 0.1
        elif key.char == '6':  # 逆时针旋转
            dyaw += 0.1
        elif key.char == '7':  # 顺时针旋转
            dyaw -= 0.1
        elif key.char == '0':  # 顺时针旋转
            vx = 0.
            vy = 0.
            dyaw = 0.
        # 限制速度范围
        vx = np.clip(vx, -1.0, 2.0)
        vy = np.clip(vy, -1.0, 0.6)
        dyaw = np.clip(dyaw, -1.0, 1.0)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()


def run_mujoco(policy, cfg):

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    default_pos = [0, 0, 0.4, -0.8, 0.4, 0, 0, 0.4, -0.8, 0.4]
    data.qpos[7:17] = default_pos
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        force = [0, 0, 0]
        cmd = np.array([[vx, vy, dyaw]], dtype=np.float32)
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.6)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.6)
            obs[0, 2:5] = cmd[0]
            obs[0, 5:15] = (q - default_pos)
            obs[0, 15:25] = dq * 0.05
            obs[0, 25:35] = action
            obs[0, 35:38] = omega
            obs[0, 38:40] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs: (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale
            target_q[0] = 0
            target_q[5] = 0
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        tau = pd_control(default_pos, target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau
        robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          name="pelvis")
        data.xfrc_applied[robot_body_id, :3] += force
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(x02Cfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/X02Lite/X02Lite.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = 1 * np.array([160, 200, 200, 200, 30, 160, 200, 200, 200, 30], dtype=np.double)
            kds = np.array([4, 5, 5, 5, 1, 4, 5, 5, 5, 1], dtype=np.double)
            tau_limit = 100. * np.ones(10, dtype=np.double)

    model_path = "../logs/x02_ppo/exported/policies/policy_1.pt"
    policy = torch.jit.load(model_path)
    run_mujoco(policy, Sim2simCfg())

