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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
class x02Cfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 40
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 52
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 0.9
        vel_limit = 0.9
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X02Lite/X02Lite.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X02Lite/X02Lite.xml'
        name = "X02Lite"
        foot_name = "ankle"
        knee_name = "knee"

        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["pelvis", "L_knee_Link", "R_knee_Link", "L_hip_pitch_Link", "R_hip_pitch_Link"]
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 1.    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            ang_vel = 0.3
            lin_vel = 0.05
            quat = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 1.08]
        pos = [0.0, 0.0, 0.95]

        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     'L_hip_yaw': 0.,
        #     'L_hip_roll': 0.,
        #     'L_hip_pitch': 0.5,
        #     'L_knee_pitch': -1.0,
        #     'L_ankle_pitch': 0.5,
        #     'R_hip_yaw': 0.,
        #     'R_hip_roll': 0.,
        #     'R_hip_pitch': 0.5,
        #     'R_knee_pitch': -1.0,
        #     'R_ankle_pitch': 0.5,
        # }

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'L_hip_yaw': 0.,
            'L_hip_roll': 0.,
            'L_hip_pitch': 0.4,
            'L_knee_pitch': -0.8,
            'L_ankle_pitch': 0.4,
            'R_hip_yaw': 0.,
            'R_hip_roll': 0.,
            'R_hip_pitch': 0.4,
            'R_knee_pitch': -0.8,
            'R_ankle_pitch': 0.4,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 160.0, 'hip_roll': 200.0, 'hip_pitch': 200.0,
                     'knee': 200.0, 'ankle': 30}
        damping = {'hip_yaw': 4, 'hip_roll': 5, 'hip_pitch': 5, 'knee': 5, 'ankle': 1}

        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.3
        disturbance = False
        disturbance_range = [-10.0, 10.0]
        disturbance_interval = 3
        dynamic_randomization = 0.02

        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        randomize_all_com = False
        rd_com_range = [-0.03, 0.03]
        randomize_base_com = True
        added_com_range = [-0.10, 0.10]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]


    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-1., 1.]

    class rewards:
        # base_height_target = 0.96
        base_height_target = 0.89
        min_dist = 0.13
        max_dist = 0.45
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.12    # rad
        target_feet_height = 0.03       # m
        cycle_time = 0.6                # sec
        target_air_time = 0.3
        stand_still = True
        run = False
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.5
        max_contact_force = 400  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            feet_contact = 1.2   # 1.2
            no_fly = 1.
            feet_contact_time = 0.
            feet_height =  1.
            feet_air_time = 1.
            foot_slip = -0.05
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.2
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            hip_pos = 0.5
            default_joint_pos = -0.01
            joint_regularization = 0.0
            ankle_regularization = 0.1
            orientation = 1.
            base_height = 0.5
            base_acc = 0.2
            foot_mirror_up = -1.
            # energy
            action_smoothness = -0.01
            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 4.
        clip_actions = 4.


class x02CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.995
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 50  # per iteration
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'x02_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
