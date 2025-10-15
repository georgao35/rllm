import yaml
import os
import json
import importlib
import random

import numpy as np
import pandas as pd
from alfworld.agents.environment import get_environment

from rllm.data.dataset import DatasetRegistry


def load_config(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def prepare_alftw_data(eval_ood=False, config_file="rllm/environments/alfworld/configs/config_tw.yaml"):
    config = load_config(config_file)

    # train alfred tw
    train_env = get_environment(env_type = config['env']['type'])(config, train_eval='train')
    train_game_files = train_env.game_files
    # train alfred tw
    eval_in_distribution_env = get_environment(env_type = config['env']['type'])(config, train_eval='eval_in_distribution')
    eval_in_distribution_game_files = eval_in_distribution_env.game_files
    # train alfred tw
    eval_ood_env = get_environment(env_type = config['env']['type'])(config, train_eval='eval_out_of_distribution')
    eval_ood_game_files = eval_ood_env.game_files

    def make_map_fn(split):
        def process_fn(game_file, idx):
            json_path = game_file.replace('game.tw-pddl', 'traj_data.json')
            with open(json_path, 'r') as f:
                traj_data = json.load(f)
            return {
                "data_source": "AlfTW",
                "game_file": game_file,
                "task_type": traj_data["task_type"],
                "train_eval": split,
                "prompt": [
                    {
                        "role": "user",
                        "content": "",  # placeholder since there is no real prompt is needed to environment based trajectory collection
                    }
                ],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": idx},
            }
        return process_fn

    train_data = [make_map_fn("train")(game_file, idx) for idx, game_file in enumerate(train_game_files)]
    eval_id_data = [make_map_fn("eval_in_distribution")(game_file, idx) for idx, game_file in enumerate(eval_in_distribution_game_files)]
    eval_ood_data = [make_map_fn("eval_out_of_distribution")(game_file, idx) for idx, game_file in enumerate(eval_ood_game_files)]

    train_dataset = DatasetRegistry.register_dataset("alftw", train_data, "train")
    eval_id_dataset = DatasetRegistry.register_dataset("alftw", eval_id_data, "eval_in_distribution")
    eval_ood_dataset = DatasetRegistry.register_dataset("alftw", eval_ood_data, "eval_out_of_distribution")

    return train_dataset, eval_ood_dataset if eval_ood else eval_id_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_alftw_data()

