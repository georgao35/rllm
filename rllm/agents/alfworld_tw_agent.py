import re
import logging
import copy
from math import inf
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


class ALFWORLDTWAgent(BaseAgent):
    SYSTEM_PROMPT: str="""
Your are an helpful expert agent operating in a household scene in the ALFRED Embodied Environment to successfully finish a particular task.
You will be provided with a description of your surroundings, including all the objects in the scene, as well as your task to accomplish.
At each step, you will receive a current observation of the scene and admissible actions. You should first reason about the situation and how to accomplish the task. Then choose an admissible action for the current step. 
""".lstrip()

    USER_PROMPT_TEMPLATE: str="""
You are an expert agent operating in the ALFRED Embodied Environment.{task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".lstrip()

    USER_PROMPT_TEMPLATE_HIST: str="""
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".lstrip()

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.hist_prompt = kwargs.get("hist_prompt", False)
        self.hist_len = kwargs.get("hist_len", inf)
        self.action_history: list[str] = []

        self.step = 0
        self.messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            }
        ]
        self._trajectory = Trajectory()
        self.task_description = ""
        self.current_obs = None
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        user_prompt = self._parse_env_update(observation, info)
        self.messages.append({"role": "user", "content": user_prompt})
        logger.debug(f"at step {self.step}, {observation=}, {reward=}, user prompt={user_prompt}")

    def update_from_model(self, response: str, **kwargs) -> Action:
        action_str, thought = self._parse_model_response(response)

        self.messages.append({"role": "assistant", "content": response})

        self._trajectory.steps.append(Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action_str,
            thought=thought,
            model_response=response,
            observation=self.current_obs,
            info={"admissible_actions": self.admissible_commands}
        ))
        self.step += 1

        action_history_str = action_str if action_str != response else "Invalid Action"
        self.action_history.append(action_history_str)
        logger.debug(f"at step {self.step}, action={action_str}")

        return Action(action=action_str)

    def _parse_env_update(self, obs, infos):
        self.admissible_commands = infos['admissible_commands']

        if len(self._trajectory.steps) == 0:
            match = re.search(r'-=.*?=-\s*([\s\S]*?)Your task is to:\s*([\s\S]*)', obs)
            self.current_obs = match.group(1).strip()
            self.task_description = match.group(2).strip()
            logger.info(f"{self.task_description=}")
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                current_observation=self.current_obs,
                task_description=f"Your task is to: {self.task_description}",
                admissible_actions=self.admissible_commands
            )
        elif self.hist_prompt:
            self.current_obs = obs
            user_prompt = self.USER_PROMPT_TEMPLATE_HIST.format(
                task_description=self.task_description,
                step_count=self.step,
                history_length=min(self.step, self.hist_len),
                action_history=self.action_history,
                current_step=self.step+1,
                current_observation=obs,
                admissible_actions=self.admissible_commands
            )
        else:
            self.current_obs = obs
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                current_observation=obs,
                task_description="",
                admissible_actions=self.admissible_commands
            )

        return user_prompt

    def _parse_model_response(self, content):
        if content.count("</think>") == 1:
            thought, sep, action_str = content.partition("</think>")
            thought = thought + sep
        else:
            thought = ""
            action_str = content

        act_match = re.search(r'<action>([\s\S]*?)</action>', action_str)
        if act_match:
            action_str = act_match.group(1).strip()
        return action_str, thought

    def get_current_state(self) -> Step | None:
        return super().get_current_state()

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            }
        ]
        self.step = 0
        self.task_description = ""
        self.action_history = []
