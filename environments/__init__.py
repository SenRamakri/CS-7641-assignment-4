import gym
from gym.envs.registration import register

from .frozen_lake import *
from .valueline import *

register(
    id='RewardingFrozenLake-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='RewardingFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8'}
)

register(
    id='RewardingFrozenLakeNoRewards16x16-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '16x16', 'rewarding': False}
)

register(
    id='RewardingFrozenLakeNoRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='RewardingFrozenLakeNoRewards8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='ValueLine-v0',
    entry_point='environments:ValueLineEnv',
)

def get_value_line_environment():
    return gym.make('ValueLine-v0')

def get_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake8x8-v0')

def get_frozen_lake_environment():
    return gym.make('FrozenLake-v0')

def get_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards8x8-v0')

def get_medium_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards16x16-v0')

def get_large_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')
