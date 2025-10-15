from cgitb import text
import os
import json
import yaml
from typing import Any
import logging
import threading

import textworld
import textworld.gym
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)

_RESET_LOCK = threading.Lock()


def load_config(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


class ALFTWEnv(BaseEnv):
    """
    adapted ALFRED TextWorld Environment that takes in game file as initialization for different tasks
    """
    def __init__(self, game_file, **kwargs) -> None:
        self.game_file = game_file
        logger.info(f"{self.game_file=}")
        ## init env
        domain_randomization = kwargs.get("domain_randomization", False)
        train_eval = kwargs.get("train_eval", "train")
        if train_eval != "train":
            domain_randomization = False
        wrappers = [AlfredDemangler(shuffle=domain_randomization), AlfredInfos]
        
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True)

        # 3. 在调用 register_games 之前获取锁
        env_id = textworld.gym.register_games(
            [self.game_file], request_infos,
            batch_size=1,
            asynchronous=True,
            wrappers=wrappers
        )
        self.env = textworld.gym.make(env_id)

    def reset(self) -> tuple[dict, dict]:
        logger.debug("Resetting ALFTWEnv")
        with _RESET_LOCK:
            obs, infos = self.env.reset()
        logger.debug(f"Reset done. {obs=}, {infos=}")
        for k in infos.keys():
            infos[k] = infos[k][0]
        return obs[0], infos

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        actions = [action]
        logger.debug(f"received {action=}")
        with _RESET_LOCK:
            obs, scores, dones, infos = self.env.step(actions)
        for k in infos.keys():
            infos[k] = infos[k][0]

        reward = float(infos['won'])
        return obs[0], reward, dones[0], infos

    def close(self):
        return self.env.close()

    @staticmethod
    def from_dict(info: dict) -> "ALFTWEnv":
        logger.info(f"Creating ALFTWEnv from dict: {info}")
        return ALFTWEnv(game_file=info["game_file"], train_eval=info.get("train_eval", "train"))


class ALFWORLDEnv(BaseEnv):

    def __init__(self, alf_config_path, **kwargs):
        config = load_config(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval=kwargs.get("train_eval", "train"))
        self.env = base_env.init_env(batch_size=1)
        print(self.env)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.admissible_commands = None

    def reset(self) -> tuple[dict, dict]:
        logger.info("Resetting ALFWORLDEnv")
        obs, infos = self.env.reset()
        for k in infos.keys():
            infos[k] = infos[k][0]
        # self.admissible_commands = infos['admissible_commands']
        return obs[0], infos

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        actions = [action]
        logging.debug(actions)
        obs, scores, dones, infos = self.env.step(actions)
        for k in infos.keys():
            infos[k] = infos[k][0]

        reward = float(infos['won'])
        if self.multi_modal:
            reward += float(infos['goal_condition_success_rate'])
        # self.admissible_commands = infos['admissible_commands']
        return obs[0], reward, dones[0], infos

    def close(self):
        return self.env.close()

    @staticmethod
    def from_dict(info: dict) -> "ALFWORLDEnv":
        return ALFWORLDEnv(alf_config_path=info["alf_config_path"], train_eval=info.get("train_eval", "train"))
