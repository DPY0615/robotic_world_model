import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
	id="Template-Isaac-Velocity-Flat-Lite3-Init-v0",
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_INIT",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg",
	},
)
gym.register(
    id="Isaac-Velocity-Flat-Lite3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPOPlayRunnerCfg",
    },
)
gym.register(
	id="Template-Isaac-Velocity-Flat-Lite3-Pretrain-v0",
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_PRETRAIN",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPOPretrainRunnerCfg",
	},
)

gym.register(
	id="Template-Isaac-Velocity-Flat-Lite3-Finetune-v0",
	entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.lite3.envs.lite3_manager_based_mbrl_env:Lite3ManagerBasedMBRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_FINETUNE",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPOFinetuneRunnerCfg",
	},
)

gym.register(
	id="Template-Isaac-Velocity-Flat-Lite3-Finetune-GlobalBalanced-v0",
	entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.lite3.envs.lite3_manager_based_mbrl_env:Lite3ManagerBasedMBRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_FINETUNE_GLOBAL_BALANCED",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPOFinetuneRunnerCfg",
	},
)

gym.register(
	id="Template-Isaac-Velocity-Flat-Lite3-Finetune-ConsYaw-v0",
	entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.lite3.envs.lite3_manager_based_mbrl_env:Lite3ManagerBasedMBRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_FINETUNE_CONS_YAW",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPOFinetuneRunnerCfg",
	},
)

gym.register(
	id="Template-Isaac-Velocity-Flat-Lite3-Visualize-v0",
	entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.lite3.envs.lite3_manager_based_mbrl_visualize_env:Lite3ManagerBasedVisualizeEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg_VISUALIZE",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPOVisualizeRunnerCfg",
	},
)
