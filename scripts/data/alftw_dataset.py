import argparse
from multiprocessing import process
import os
import json
import yaml

import numpy as np
import pandas as pd
from alfworld.agents.environment import get_environment

import rllm
from rllm.data.dataset import DatasetRegistry

# Get the directory for rLLM repo (rllm.__file__)
RLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__)))


def load_config(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("config_file")
    parser.add_argument("--local_dir", default=os.path.join(RLLM_DIR, "data/rllm-alfworld"))
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    config_file = args.config_file
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

    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    eval_id_df = pd.DataFrame(eval_id_data)
    eval_id_df.to_parquet(os.path.join(local_dir, "eval_in_distribution.parquet"))
    eval_ood_df = pd.DataFrame(eval_ood_data)
    eval_ood_df.to_parquet(os.path.join(local_dir, "eval_out_of_distribution.parquet"))

    train_dataset = DatasetRegistry.register_dataset("alftw", train_data, "train")
    eval_id_dataset = DatasetRegistry.register_dataset("alftw", eval_id_data, "eval_in_distribution")
    eval_ood_dataset = DatasetRegistry.register_dataset("alftw", eval_ood_data, "eval_out_of_distribution")


if __name__ == "__main__":
    main()
