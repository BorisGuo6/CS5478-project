import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    rl_task_env = task_registry.make_task("navigation_task", headless=False, num_envs=16)
    rl_task_env.reset()
    actions = torch.zeros(
        (rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)
    ).to("cuda:0")
    actions[:, 0] = -1.0
    logger.info(
        "\n\n\n\n\n\n This script provides an example of the RL task interface with a zero action command in a cluttered environment."
    )
    logger.info(
        "This is to indicate the kind of interface that is available to the RL algorithm and the users for interacting with a Task environment.\n\n\n\n\n"
    )
    rl_task_env.sim_env.reset()
    num_assets_in_env = (
        rl_task_env.sim_env.IGE_env.num_assets_per_env - 1
    )  # subtract 1 because the robot is also an asset
    logger.info(f"Number of assets in the environment: {num_assets_in_env}")
    num_envs = rl_task_env.task_config.num_envs

    asset_twist = torch.zeros(
        (num_envs, num_assets_in_env, 6), device="cuda:0", requires_grad=False
    )
    
    asset_twist[:, :, 0] = -1.0
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            # asset_twist[:, :, 0] = torch.sin(0.2 * i * torch.ones_like(asset_twist[:, :, 0]))
            # asset_twist[:, :, 1] = torch.cos(0.2 * i * torch.ones_like(asset_twist[:, :, 1]))
            # asset_twist[:, :, 2] = 0.0
            # rl_task_env.sim_env.step(actions=actions, env_actions=asset_twist)
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
    end = time.time()
