set -x

export N_PARALLEL_AGENTS=16
export K=4
export MODEL_NAME=Qwen/Qwen3-1.7B
export EVAL_OOD="true"

python examples/alftw/run_alfred_tw_agent.py


export N_PARALLEL_AGENTS=16
export K=4
export MODEL_NAME=Qwen/Qwen3-1.7B
export EVAL_OOD="false"

python examples/alftw/run_alfred_tw_agent.py

