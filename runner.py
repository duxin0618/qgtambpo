import tensorflow as tf
from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import set_seed, initialize_tf_variables

import static
import socket
import time
from shutil import copyfile
import os

tf.logging.set_verbosity(tf.logging.ERROR)
class ExperimentRunner:
    def __init__(self, variant):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.variant = variant
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU90%的显存
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()
        self.train_generator = None

    def build(self):
        environment_params = self.variant['environment_params']
        dir_path_params = self.variant['dir_path']
        training_environment = self.training_environment = (get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (get_replay_pool_from_variant(self.variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(self.variant)

        Qs = self.Qs = get_Q_function_from_variant(self.variant, training_environment)
        policy = self.policy = get_policy_from_variant(self.variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (get_policy('UniformPolicy', training_environment))

        domain = environment_params['training']['domain']
        static_fns = static[domain.lower()]

        tag = time.time()
        env_name = self.variant['environment_params']['training']['domain']
        log_dir = dir_path_params['log_dir']
        if not os.path.exists(log_dir+'%s' % env_name):
            os.makedirs(log_dir+'%s' % env_name)
        diagnostics_file = log_dir + '{}/diagnostics_{}_{}.csv'.format(env_name, env_name, tag)

        source_file = 'config/%s.py' % env_name.lower()
        target_file = log_dir+'%s/%d.config' % (env_name, tag)

        copyfile(source_file, target_file)
        with open(target_file, 'a') as f_config:
            f_config.write('\n')
            f_config.write(socket.gethostname())

        self.algorithm = get_algorithm_from_variant(
            variant=self.variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            static_fns=static_fns,
            sampler=sampler,
            session=self._session,
            log_file=log_dir+'%s/test_%d.log' % (self.variant['algorithm_params']['domain'], tag),
            env_name=env_name,
            diagnostics_file=diagnostics_file,
            tag=tag)

        initialize_tf_variables(self._session, only_uninitialized=True)

    def train(self):
        self.build()

        self.algorithm.train()
