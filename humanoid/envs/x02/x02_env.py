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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from humanoid.envs import LeggedRobot
from humanoid.utils.terrain import HumanoidTerrain
# from collections import deque


class x02Env(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # self.last_feet_z = 0.10
        self.last_feet_z = 0.06
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_contact_history = torch.zeros(self.num_envs, 30, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.max_feet_air_time = torch.zeros_like(self.feet_air_time)
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)
        self.feet_in_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.phase = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_phase = torch.zeros((self.num_envs, 2), device=self.device)

        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                       requires_grad=False)

        # self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        # self.compute_observations()

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def create_sim(self):
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 15] =  noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[15: 25] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[25: 35] = 0.  # previous actions
        noise_vec[35: 38] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[38: 40] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def _get_contact_mask(self):
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        return contact_mask

    def _get_first_contact_mask(self):
        contacts = self._get_contact_mask()
        first_contact = torch.logical_and(contacts, ~self.last_contacts)
        # first_contact = torch.sum(first_contact, dim=1)
        return first_contact

    def _get_lift_mask(self):
        contacts = self._get_contact_mask()
        lift = torch.logical_and(~contacts, self.last_contacts)
        # lift = torch.sum(lift, dim=1)
        return lift

    def _get_walk_mask(self):
        self.feet_phase[:, 0] = self.phase[:, 0]
        self.feet_phase[:, 1] = torch.fmod(self.phase[:, 0] + 0.5, 1.0)
        walk_mask = self.feet_phase < 0.5
        return walk_mask

    def update_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        dt_phase = self.dt / cycle_time
        self.phase = (self.phase + dt_phase) % 1.0

    def compute_phase(self):
        phase_sin = torch.sin(2 * torch.pi * self.phase)
        phase_cos = torch.cos(2 * torch.pi * self.phase)
        sc = torch.cat((phase_sin, phase_cos), dim=1)
        return sc

    def step(self, actions):
        # dynamic randomization
        # delay = torch.rand((self.num_envs, 1), device=self.device)
        # actions = (1 - delay) * actions + delay * self.actions
        # actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations(self):
        contact_mask = self._get_contact_mask()
        walk_mask = self._get_walk_mask()

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        sc = self.compute_phase()

        self.privileged_obs_buf = torch.cat((
            sc,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_euler_xyz[:,:2] * self.obs_scales.quat,
            self.rand_push_force[:, :2],
            self.rand_push_torque,
            walk_mask,
            contact_mask,
        ), dim=-1)

        obs_buf = torch.cat((
            sc,
            self.commands[:, :3] * self.commands_scale,
            q,
            dq,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_euler_xyz[:,:2]* self.obs_scales.quat,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() +(2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        self.feet_contact_time[env_ids] = 0
        self.max_feet_air_time[env_ids, :] = 0

        self.phase[env_ids, 0] = torch.rand((torch.numel(env_ids),), requires_grad=False, device=self.device)

        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0



    def post_physics_step(self):
        # ---feet observation--
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.feet_contact_filt = torch.logical_or(contacts, self.last_contacts)
        self.last_contacts = contacts
        self.feet_contact_history = torch.cat([self.feet_contact_history[:, 1:].clone(), self.feet_contact_filt[:, None, :].clone()], dim=1)
        super().post_physics_step()

    def _post_physics_step_callback(self):
        self.update_phase()
        super()._post_physics_step_callback()

    # ================================================ Rewards ================================================== #

    #####################     foot-pos    #################################################################
    def _reward_feet_contact(self):
        walk_mask = self._get_walk_mask()
        reward = torch.where(self.feet_contact_filt == walk_mask, 1., -0.3)
        reward = torch.mean(reward, dim=1)
        return reward

    # def _reward_no_fly(self):
    #     contact = self._get_contact_mask()
    #     single_contact = torch.sum(contact, dim=1) == 1
    #     reward = 1. * single_contact
    #     return reward

    def _reward_no_fly(self):
        # reward one foot touch plane once at least in 0.2s
        single_contact = torch.sum(self.feet_contact_history, dim=2) == 1
        rew_contact = torch.sum(1 * single_contact, dim=1) >= 20
        reward = 1. * rew_contact
        return reward

    def _reward_feet_contact_time(self):
        lift = self._get_lift_mask()
        single_lift = torch.sum(lift, dim=1) == 1
        first_contact = self._get_first_contact_mask()
        single_first_contact = torch.sum(first_contact, dim=1) == 1

        self.feet_in_contact = torch.where(single_first_contact, True, self.feet_in_contact)
        self.feet_in_contact = torch.where(single_lift, False, self.feet_in_contact)
        self.feet_contact_time += self.dt
        self.feet_contact_time *= self.feet_contact_filt
        feet_contact_time = torch.sum(self.feet_contact_time, dim=1)
        feet_contact_time = feet_contact_time.clamp(0, 0.5)
        rew_mask = torch.abs(feet_contact_time - 0.3) < 0.1
        reward = 1. * rew_mask * self.feet_in_contact
        return reward

    def _reward_feet_air_time(self):
        lift = self._get_lift_mask()
        single_lift = torch.sum(lift, dim=1) == 1
        first_contact = self._get_first_contact_mask()
        single_first_contact = torch.sum(first_contact, dim=1) == 1

        self.feet_in_air = torch.where(single_lift, True, self.feet_in_air)
        self.feet_in_air = torch.where(single_first_contact, False, self.feet_in_air)
        self.feet_air_time += self.dt
        feet_air_time = torch.sum((self.feet_air_time - 0.3), dim=1) * self.feet_in_air
        self.feet_air_time *= ~self.feet_contact_filt
        self.max_feet_air_time = torch.where(self.max_feet_air_time > self.feet_air_time, self.max_feet_air_time, self.feet_air_time)
        error = self.max_feet_air_time[:, 0] - self.max_feet_air_time[:, 1]
        reward = feet_air_time - error
        return reward

    def _reward_feet_height(self):
        lift = self._get_lift_mask()
        single_lift = torch.sum(lift, dim=1) == 1
        first_contact = self._get_first_contact_mask()
        single_first_contact = torch.sum(first_contact, dim=1) == 1

        self.feet_in_air = torch.where(single_lift, True, self.feet_in_air)
        self.feet_in_air = torch.where(single_first_contact, False, self.feet_in_air)
        # feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.10
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.06
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.feet_height *= ~self.feet_contact_filt
        feet_height = torch.sum(self.feet_height, dim=1) * self.feet_in_air
        self.last_feet_z = feet_z
        rew_mask = torch.abs(feet_height - self.cfg.rewards.target_feet_height) < 0.01
        reward = 1. * rew_mask
        return reward

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:10], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        reward = torch.sum(rew, dim=1)
        return reward

    def _reward_feet_contact_forces(self):
        error = torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)
        return error

    #####################    Orientation     ###################################################################

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        error = torch.abs(base_height - self.cfg.rewards.base_height_target)
        reward = torch.exp(-10. * error / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        reward = torch.norm(joint_diff, dim=1)
        return reward

    def _reward_hip_pos(self):
        base_lin_vel_y = torch.abs(self.commands[:, 1]) > 0.1
        joint_diff = self.dof_pos - self.default_joint_pd_target
        hip_yaw_indices = [0, 5]
        hip_roll_indices = [1, 6]
        hip_yaw_error = joint_diff[:, hip_yaw_indices]
        hip_roll_error = joint_diff[:, hip_roll_indices]
        error1 = torch.norm(hip_yaw_error, dim=1)
        error2 = torch.norm(hip_roll_error, dim=1)
        error2 = torch.where(base_lin_vel_y, error2 * 0.3, error2)
        error = error1 + error2
        error = torch.clamp(error - 0.1, 0, 50)
        reward = torch.exp(-error * 100)
        return reward

    def _reward_foot_mirror_up(self):
        error = torch.sum(torch.square(self.dof_pos[:, [0, 1, 2, 3, 4]] - self.dof_pos[:, [5, 6, 7, 8, 9]]), dim=-1)
        return error *  torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        reward = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return reward

    #######################     vel     ######################################################################################

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        reward = (lin_mismatch + ang_mismatch) / 2.
        return reward

    def _reward_track_vel_hard(self):
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        reward = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
        return reward

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square( self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_low_speed(self):
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    #####################   Energy    ##########################################################

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_smoothness(self):
        error_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        error_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        error = error_1 + error_2
        return error

    #####################   PB  ########################################################################

    def _reward_joint_regularization(self):
        error = 0.
        error += self.sqrdexp(
            1. * (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            1. * (self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            1. * (self.dof_pos[:, 1]) + self.dof_pos[:, 6] / self.cfg.normalization.obs_scales.dof_pos)
        return error

    def _reward_ankle_regularization(self):
        error = 0
        error += self.sqrdexp(
            1. * ((self.dof_pos[:, 4]) - self.default_dof_pos[:, 4]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            1. * ((self.dof_pos[:, 9]) - self.default_dof_pos[:, 9]) / self.cfg.normalization.obs_scales.dof_pos)
        return error


# ##################### HELPER FUNCTIONS ################################## #
    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * 1.
        return torch.sin(p) / (2*torch.sqrt(torch.sin(p)**2. + 0.2**2.)) + 1./2.



# python scripts/train.py --task=x02_ppo --run_name v1 --headless --num_envs 4096
# python scripts/play.py --task=x02_ppo --run_name v1 --num_envs 64
# python scripts/sim2sim.py --load_model ../logs/x02_ppo/exported/policies/policy_1.pt


