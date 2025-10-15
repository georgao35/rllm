import asyncio
import hydra
import logging

from transformers import AutoTokenizer

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.utils import compute_pass_at_k, save_trajectories
from rllm.agents.alfworld_tw_agent import ALFWORLDTWAgent
from rllm.environments.alfworld.alfworld import ALFTWEnv

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_alftw_data(eval_ood=False):
    if DatasetRegistry.dataset_exists("alftw", "eval_in_distribution"):
        test_dataset = DatasetRegistry.load_dataset("alftw", "eval_in_distribution")
        return test_dataset.get_data()

    print("FrozenLake datasets not found. Preparing datasets...")
    from prepare_alftw_data import prepare_alftw_data

    train_dataset, test_dataset = prepare_alftw_data(eval_ood=eval_ood, config_file="rllm/environments/alfworld/configs/config_tw.yaml")

    return test_dataset.get_data()


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 4

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    agent_args = {
        "max_steps": 50,
        "use_accumulate_history": True,
    }

    env_args = {
        "seed": 47,
        "max_steps": 8,
    }

    engine = AgentExecutionEngine(
        agent_class=ALFWORLDTWAgent,
        env_class=ALFTWEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:12345/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
        max_steps=20
    )

    eval_ood = True
    tasks = load_alftw_data(eval_ood)

    results = asyncio.run(engine.execute_tasks(tasks[:10]*2))
    compute_pass_at_k(results)
    save_trajectories(results, filename=f"alfred_tw_trajectories-{'ood' if eval_ood else 'id'}.jsonl")


if __name__ == "__main__":
    import cProfile
    import pstats
    
    cProfile.run('main()', 'profile_output.prof')
    stats = pstats.Stats('profile_output.prof')
    stats.sort_stats('cumulative').print_stats(10) 
