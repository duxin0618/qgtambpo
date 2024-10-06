import sys

from .ant import AntEnv
from .swimmer import SwimmerEnv
from .humanoid import HumanoidTruncatedObsEnv

env_overwrite = {'Ant': AntEnv, 'Swimmer': SwimmerEnv, 'Humanoid': HumanoidTruncatedObsEnv}

sys.modules[__name__] = env_overwrite