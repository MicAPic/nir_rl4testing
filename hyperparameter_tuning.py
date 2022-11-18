from ray import tune
from utility import RetentionReplayBufferRay
# import numpy as np
# from ray.rllib.algorithms.dqn import DQN

results = tune.run(
    'DQN',
    stop={
        'timesteps_total': 50_000
    },
    config={
        "env": 'LunarLander-v2',
        "num_workers": 2,
        "hiddens": [64, 64],
        "dueling": False,
        "double_q": False,
        "gamma": tune.grid_search([0.999, 0.8]),
        "lr": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5]),
        "replay_buffer_config": {
            "type": tune.grid_search(["ReplayBuffer", RetentionReplayBufferRay]),
            "capacity": tune.grid_search([1000000, 10000000, 100000000]),
        },
    }
)
#
# config = {
#     "env": 'LunarLander-v2',
#     'gamma': 0.999,
#     'lr': 1e-4,
#
#     "rollout_fragment_length": 4,
#     "train_batch_size": 32,
#
#     "replay_buffer_config": {
#         "type": RetentionReplayBuffer,
#         "capacity": 10_000_000,
#     },
#
#     "num_workers": 2,
#     "framework": "tf",
#     "exploration_config": {
#         "epsilon_timesteps": 200_000,
#         "final_epsilon": 0.01,
#     },
#
#     "model": {
#         "fcnet_hiddens": [64, 64],
#         "fcnet_activation": "relu",
#     },
#
#     "evaluation_num_workers": 1,
#     "evaluation_config": {
#         "render_env": True,
#     },
# }
#
# model = DQN(config=config)
#
# scores, avg_scores, episodes = [], [], [0.0]
# score, avg_score, i = 0.0, 0.0, 0
# for i in range(100):
# # while avg_score < 200.0:
#     step = model.train()
#     scores.extend(step['hist_stats']['episode_reward'])
#     avg_score = np.mean(scores[-100:])
#     avg_scores.append(avg_score)
#     episodes.append(len(avg_scores))
#     print(f"{i}) Sampled steps: {step['num_env_steps_sampled']}, average score: {avg_score}")
#     i += 1
#
# # model.evaluate()
