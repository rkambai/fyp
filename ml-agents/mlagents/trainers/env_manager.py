from abc import ABC, abstractmethod

from typing import List, Dict, NamedTuple, Iterable, Tuple
from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    BehaviorName,
)
from mlagents_envs.side_channel.stats_side_channel import EnvironmentStats

from mlagents.trainers.policy import Policy
from mlagents.trainers.agent_processor import AgentManager, AgentManagerQueue
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.logging_util import get_logger

from mlagents.trainers.CLIPEncoderBase import CLIPEncoderBase
from mlagents.trainers.cli_utils import load_config

AllStepResult = Dict[BehaviorName, Tuple[DecisionSteps, TerminalSteps]]
AllGroupSpec = Dict[BehaviorName, BehaviorSpec]
FYP_CONFIG_PATH = r"C:\Users\Rainer\fyp\ml-agents\ml-agents\mlagents\fyp_config.yml"
# FYP_CONFIG_PATH = r"/content/fyp/ml-agents/mlagents/fyp_config.yml"

logger = get_logger(__name__)


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    worker_id: int
    brain_name_to_action_info: Dict[BehaviorName, ActionInfo]
    environment_stats: EnvironmentStats

    @property
    def name_behavior_ids(self) -> Iterable[BehaviorName]:
        return self.current_all_step_result.keys()

    @staticmethod
    def empty(worker_id: int) -> "EnvironmentStep":
        return EnvironmentStep({}, worker_id, {}, {})


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[BehaviorName, Policy] = {}
        self.agent_managers: Dict[BehaviorName, AgentManager] = {}
        self.first_step_infos: List[EnvironmentStep] = []

        ## ADD CLIP ENCODER
        self.fyp_config = load_config(FYP_CONFIG_PATH)
        self.clip_config = self.fyp_config['CLIP']
        self.encoder = CLIPEncoderBase(self.clip_config['model_path'], self.clip_config['processor_path'])
        self.encoder.start_session()
        self.prompt = self.fyp_config['target_enum'][0]['prompt']
        self.next_lesson_measure = self.fyp_config['lesson_threshold'] * self.fyp_config['max_steps']
        self.use_clip = self.fyp_config['use_clip']
        self.global_step_count = 0

    def set_policy(self, brain_name: BehaviorName, policy: Policy) -> None:
        self.policies[brain_name] = policy
        if brain_name in self.agent_managers:
            self.agent_managers[brain_name].policy = policy

    def set_agent_manager(
        self, brain_name: BehaviorName, manager: AgentManager
    ) -> None:
        self.agent_managers[brain_name] = manager

    @abstractmethod
    def _step(self) -> List[EnvironmentStep]:
        pass

    @abstractmethod
    def _reset_env(self, config: Dict = None) -> List[EnvironmentStep]:
        pass

    def reset(self, config: Dict = None) -> int:
        for manager in self.agent_managers.values():
            manager.end_episode()
        # Save the first step infos, after the reset.
        # They will be processed on the first advance().
        self.first_step_infos = self._reset_env(config)
        return len(self.first_step_infos)

    @abstractmethod
    def set_env_parameters(self, config: Dict = None) -> None:
        """
        Sends environment parameter settings to C# via the
        EnvironmentParametersSideChannel.
        :param config: Dict of environment parameter keys and values
        """
        pass

    def on_training_started(
        self, behavior_name: str, trainer_settings: TrainerSettings
    ) -> None:
        """
        Handle traing starting for a new behavior type. Generally nothing is necessary here.
        :param behavior_name:
        :param trainer_settings:
        :return:
        """
        pass

    @property
    @abstractmethod
    def training_behaviors(self) -> Dict[BehaviorName, BehaviorSpec]:
        pass

    @abstractmethod
    def close(self):
        pass

    def get_steps(self) -> List[EnvironmentStep]:
        """
        Updates the policies, steps the environments, and returns the step information from the environments.
        Calling code should pass the returned EnvironmentSteps to process_steps() after calling this.
        :return: The list of EnvironmentSteps
        """
        # If we had just reset, process the first EnvironmentSteps.
        # Note that we do it here instead of in reset() so that on the very first reset(),
        # we can create the needed AgentManagers before calling advance() and processing the EnvironmentSteps.
        if self.first_step_infos:
            self._process_step_infos(self.first_step_infos)
            self.first_step_infos = []
        # Get new policies if found. Always get the latest policy.
        for brain_name in self.agent_managers.keys():
            _policy = None
            try:
                # We make sure to empty the policy queue before continuing to produce steps.
                # This halts the trainers until the policy queue is empty.
                while True:
                    _policy = self.agent_managers[brain_name].policy_queue.get_nowait()
            except AgentManagerQueue.Empty:
                if _policy is not None:
                    self.set_policy(brain_name, _policy)
        # Step the environments
        new_step_infos = self._step()
        self.global_step_count += 1
        return new_step_infos

    def process_steps(self, new_step_infos: List[EnvironmentStep]) -> int:
        # Add to AgentProcessor
        num_step_infos = self._process_step_infos(new_step_infos)
        return num_step_infos

    def _process_step_infos(self, step_infos: List[EnvironmentStep]) -> int:

        if self.global_step_count >= self.next_lesson_measure:
            self.prompt = self.fyp_config['target_enum'][1]['prompt']
        for step_info in step_infos:
            for name_behavior_id in step_info.name_behavior_ids:
                if name_behavior_id not in self.agent_managers:
                    logger.warning(
                        "Agent manager was not created for behavior id {}.".format(
                            name_behavior_id
                        )
                    )
                    continue
                decision_steps, terminal_steps = step_info.current_all_step_result[
                    name_behavior_id
                ]

                ## ADD PROCESS HERE
                if self.use_clip:
                    decision_steps = self.CLIPProcessStep(decision_steps)

                self.agent_managers[name_behavior_id].add_experiences(
                    decision_steps,
                    terminal_steps,
                    step_info.worker_id,
                    step_info.brain_name_to_action_info.get(
                        name_behavior_id, ActionInfo.empty()
                    ),
                )

                self.agent_managers[name_behavior_id].record_environment_stats(
                    step_info.environment_stats, step_info.worker_id
                )
        return len(step_infos)
    
    def CLIPProcessStep(self, decision_steps):

        r_similarity_multiplier = 0.01

        new_rewards = decision_steps.reward
        for batch_obs in decision_steps.obs:
            for agent_idx, obs in enumerate(batch_obs):
                self.encoder.run_inference(self.prompt, obs)
                r_similarity = self.encoder.r_similarity_linear
                new_rewards[agent_idx] += (r_similarity * r_similarity_multiplier)

        new_decision_steps = DecisionSteps(
            decision_steps.obs,
            new_rewards,
            decision_steps.agent_id,
            decision_steps.action_mask,
            decision_steps.group_id,
            decision_steps.group_reward)
            
        return new_decision_steps
