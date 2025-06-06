import torch
import math
import genesis as gs
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rl_zoo3.envs.aegis.aegis_env_cfg import ENV_CFG


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class AegisPusherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, show_viewer=False, device="cuda", render_mode=None):
        super().__init__()

        if not gs._initialized:
            gs.init(logging_level="warning")

        if render_mode == "human":
            show_viewer = True

        self.render_mode = render_mode
        self.device = torch.device(device)

        self.dt = ENV_CFG["dt"]
        self.max_episode_length = math.ceil(ENV_CFG["episode_length_s"] / self.dt)

        self.num_obs = ENV_CFG["num_obs"]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
        self.num_actions = ENV_CFG["num_actions"]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.target_threshold = ENV_CFG["target_threshold"]

        self.obs_scales = ENV_CFG["obs_scales"]
        self.reward_scales = ENV_CFG["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=5),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                # enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file='rl_zoo3/envs/aegis/description/aegis.urdf', fixed=True,
                pos=ENV_CFG["robot_pos"],
            ),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.84, 0.55, 0.82),
                pos=ENV_CFG["table_pos"],
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.0, 0.7, 0.84),
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
            material=gs.materials.Rigid(rho=8000.0, friction=0.6, coup_friction=0.6),
        )

        self.scene.build()

        self.episode_step = 0.0

        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in ENV_CFG["dof_names"]]
        self.robot.set_dofs_kp(ENV_CFG["kp"], self.motor_dofs)
        self.robot.set_dofs_kv(ENV_CFG["kd"], self.motor_dofs)

        self.default_dof_pos = torch.tensor(
            [ENV_CFG["default_joint_angles"][name] for name in ENV_CFG["dof_names"]],
            device=self.device
        )
        self.dof_pos = torch.zeros(self.num_actions, device=self.device)
        self.dof_vel = torch.zeros(self.num_actions, device=self.device)
        self.tcp_pos = torch.zeros(3, device=self.device)
        self.tcp_vel = torch.zeros(3, device=self.device)
        self.object_pos = torch.zeros(3, device=self.device)
        self.target_pos = torch.tensor(ENV_CFG["target_pos"], device=self.device, dtype=torch.float32)

        self.actions = torch.zeros(self.num_actions, device=self.device)
        self.last_actions = torch.zeros(self.num_actions, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_actions, device=self.device)

        self.reward_functions = {
            "near": self._reward_near,
            "dist": self._reward_dist,
            "control": self._reward_control,
        }

        self.episode_sums = {key: 0.0 for key in self.reward_functions}

        self.reset()

    def step(self, action):
        action = np.clip(action, -ENV_CFG["clip_actions"], ENV_CFG["clip_actions"])
        self.actions.copy_(torch.from_numpy(action).to(self.device))

        target_dof_pos = self.dof_pos + self.actions * ENV_CFG["action_scale"]
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        self.scene.step()
        self.episode_step += 1

        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)
        self.tcp_pos = self.robot.get_links_pos()[7, :]
        self.tcp_vel = self.robot.get_links_vel()[7, :]
        self.object_pos = self.object.get_pos()

        dist_to_target = torch.norm(self.target_pos - self.object_pos)
        success = bool((dist_to_target < self.target_threshold).item())

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        # if success:
        #     reward += 5
        reward = float(reward.item())

        self.episode_return += reward

        terminated = bool(success)
        truncated = self.episode_step >= self.max_episode_length

        info = {
            "success": success,
            "dist_to_target": dist_to_target.item(),
            "episode_step": self.episode_step,
            "is_truncated": truncated,
            "is_success": success,
        }

        for key, value in self.episode_sums.items():
            info[f"reward_{key}"] = value.item()

        if terminated or truncated:
            info["episode"] = {
                "r": float(reward),
                "l": self.episode_step
            }

        obs = torch.cat([
            self.dof_pos * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.tcp_pos,
            self.tcp_vel,
            self.object_pos,
            self.target_pos,
        ]).cpu().numpy()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.dof_pos = self.default_dof_pos.clone()
        self.dof_vel = torch.zeros_like(self.dof_vel)
        self.robot.set_dofs_position(
            position=self.dof_pos,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()

        self.tcp_pos = self.robot.get_links_pos()[7, :].float()
        self.tcp_vel = self.robot.get_links_vel()[7, :].float()

        x_range = ENV_CFG["object_spawn_x"]
        y_range = ENV_CFG["object_spawn_y"]
        z_range = ENV_CFG["object_spawn_z"]

        rand_pos = torch.tensor([
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
            np.random.uniform(z_range[0], z_range[1]),
        ], device=self.device)

        self.object.set_pos(rand_pos, zero_velocity=True)
        self.object_pos[:] = self.object.get_pos()

        self.actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.last_dof_vel[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}

        obs = torch.cat([
            self.dof_pos * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.tcp_pos,
            self.tcp_vel,
            self.object_pos,
            self.target_pos
        ]).cpu().numpy()

        return obs, {}

    def _reward_near(self):
        return torch.norm(self.tcp_pos - self.object_pos)

    def _reward_dist(self):
        return torch.norm(self.target_pos - self.object_pos)

    def _reward_control(self):
        return torch.sum(self.actions ** 2)

    def render(self):
        pass
