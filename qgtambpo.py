import math
from collections import OrderedDict
from numbers import Number
from itertools import count
import os
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from models.constructor import construct_model, format_samples_for_training
from models.fake_env import FakeEnv


def td_target(reward, discount, next_value):
    return reward + discount * next_value

tf.logging.set_verbosity(tf.logging.ERROR)

class QGTAMBPO(RLAlgorithm):

    def __init__(
            self,
            env_name,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            log_file=None,
            diagnostics_file=None,
            plotter=None,
            tf_summaries=False,
            tag=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,

            deterministic=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            rollout_schedule=[20, 100, 1, 1],
            hidden_dim=200,
            max_model_t=None,

            use_ampo=False,
            n_adapt_per_epoch=2000,
            epoch_stop_adapt=1000,
            n_itr_critic=5,
            adapt_batch_size=256,

            po_stop_epoch=-1,
            src_min_length=1,
            use_src=False,
            src_iter_schedule=[1, 1, 1, 1],
            plan=0,
            src_fix_iter=False,
            src_epoch_stop=25,
            src_max_length=10,
            src_pool_size=10e3,
            src_min_fake_sample_size=10e3,
            Q_tau_info=[20, 100, 0.3, 0.5],
            src_reset_rollout_length=-1,

            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(QGTAMBPO, self).__init__(**kwargs)

        obs_dim = np.prod(training_environment.observation_space.shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites,
                                      adapt_batch_size=adapt_batch_size)
        self._static_fns = static_fns
        self._env_name = env_name
        self._tag = tag
        self.fake_env = FakeEnv(self._model, self._static_fns)

        self._rollout_schedule = rollout_schedule
        self._epoch_stop_adapt = epoch_stop_adapt
        self._max_model_t = max_model_t
        self._n_adapt_per_epoch = n_adapt_per_epoch
        self._n_itr_critic = n_itr_critic
        self._adapt_batch_size = adapt_batch_size
        self._po_stop_epoch = po_stop_epoch
        self._use_ampo = use_ampo

        self._src_min_length = src_min_length
        self._use_src = use_src
        self._src_iter_schedule = src_iter_schedule
        self._plan = plan
        self._src_fix_iter = src_fix_iter
        self._src_max_length = src_max_length
        self._src_epoch_stop = src_epoch_stop
        self._src_pool_size = src_pool_size
        self._src_min_fake_sample_size = src_min_fake_sample_size
        self._start_train_Q=True
        self._Q_tau_info = Q_tau_info
        self._src_reset_rollout_length = src_reset_rollout_length

        self._model_retain_epochs = model_retain_epochs
        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy
        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)
        self._Q_critic_simulation = tuple(tf.keras.models.clone_model(Q) for Q in Qs)
        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries
        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape
        self.log_file = log_file
        self.diagnostics_file = diagnostics_file
        self._build()

    def _build(self):
        self._training_ops = {}
        self._real_training_Q_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_critic_simulated_update()

    def _train(self):
        print('log file:', self.log_file)
        f_log = open(self.log_file, 'a')
        if not self._use_src:
            info = "plan MBPO"
        else:
            info = "plan {}\n".format(self._plan, self._alpha)
        print(info)
        f_log.write(info)
        min_epoch, max_epoch, min_iter, max_iter = self._rollout_schedule
        info = "rollout_schedule: [{0}, {1}, {2}, {3}] \n".format(min_epoch, max_epoch, min_iter, max_iter)
        f_log.write(info)
        f_log.close()

        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool
        model_metrics = {}

        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool)

        self.sampler.initialize(training_environment, policy, pool)
        self._training_before_hook()

        for self._epoch in range(self._epoch, self._n_epochs):
            self._epoch_before_hook()

            start_samples = self.sampler._total_samples
            print("\033[0;31m%s%d\033[0m" % ('epoch: ', self._epoch))
            print("\033[0;32m%s %d\033[0m" % ('tag: ', self._tag))
            print('[ True Env Buffer Size ]', pool.size)

            # train Q network
            if self._start_train_Q:
                self._do_training_real_q_repeat(timestep=self._total_timestep)

            for i in count():

                self._src_Q_file_index = i

                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if samples_now >= start_samples + self._epoch_length and self.ready_to_train:
                    break

                self._timestep_before_hook()
                if self._epoch >= 120:
                    self._real_ratio = 1

                if self._timestep % self._model_train_freq == 0 and self._epoch < 120:  # 250
                    # train model
                    print('[Train] Begin train the model')
                    model_train_metrics = self._train_model(batch_size=256, max_epochs=None, holdout_ratio=0.2,
                                                            max_t=self._max_model_t)
                    model_metrics.update(model_train_metrics)
                    print('Finish train the model')
                    self._set_rollout_length()

                    # model train by src function
                    if self._use_src and self._src_epoch_stop > self._epoch \
                        and self._src_max_length >= self._rollout_length \
                        and self._src_min_length <= self._rollout_length:
                        # self._model.src_copy_source_to_save()  # save source model
                        # training by src function
                        print('[src] Begin use src function train the model')
                        self._set_src_iter()
                        self._reallocate_use_src_model_pool()
                        model_by_src_function_train_metrics = self._train_model_by_src_function(batch_size=512,
                                                                                                max_epochs=None,
                                                                                                holdout_ratio=0.2,
                                                                                                max_t=self._max_model_t)
                        model_metrics = self.constructor_model_metrics(model_by_src_function_train_metrics,
                                                                       model_metrics)

                        print('Finish use src function train the model')
                        # end training by src function
                    else:
                        model_by_src_function_train_metrics = {'src_model_train_loss': 0, 'src_model_val_loss': 0}
                        model_metrics = self.constructor_model_metrics(model_by_src_function_train_metrics,
                                                                       model_metrics)
                    # rollout
                    self._reallocate_model_pool()
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size,
                                                                deterministic=self._deterministic)

                    model_metrics = self.constructor_model_metrics(model_rollout_metrics, model_metrics)

                self._do_sampling(timestep=self._total_timestep)

                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)

                self._timestep_after_hook()

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
            else:
                evaluation_metrics = {}

            self._epoch_after_hook(training_paths)

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            self.save_diagnostics(diagnostics)
            f_log = open(self.log_file, 'a')
            f_log.write('epoch: %d\n' % self._epoch)
            f_log.write('total time steps: %d\n' % self._total_timestep)
            f_log.write('rollout_length: %d\n' % self._rollout_length)
            f_log.write('evaluation return: %f\n' % evaluation_metrics['return-average'])
            print('evaluation return: %f\n' % evaluation_metrics['return-average'])
            f_log.write('current time: %s\n' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f_log.close()

            if self._epoch == self._po_stop_epoch:
                break

        self.sampler.terminate()

        self._training_after_hook()

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _set_n_adapt(self):
        if str(type(self._n_adapt_per_epoch))[-13:-2] == 'ListWrapper':
            min_epoch, max_epoch, min_adapt, max_adapt = self._n_adapt_per_epoch
            if self._epoch <= min_epoch:
                y = min_adapt
            else:
                dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
                dx = min(dx, 1)
                y = dx * (max_adapt - min_adapt) + min_adapt

            self._n_adapt = int(y)
        else:
            self._n_adapt = self._n_adapt_per_epoch

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        if self._src_epoch_stop < self._epoch and self._src_reset_rollout_length > 0:
            self._rollout_length = self._src_reset_rollout_length
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reset_Q_tau(self):
        min_epoch, max_epoch, min_qtau, max_qtau = self._Q_tau_info
        if self._epoch < min_epoch:
            cur = min_qtau
        elif self._epoch >= max_epoch:
            cur = max_qtau
        else:
            cur = min_qtau + round((max_qtau - min_qtau) / (max_epoch - min_epoch) * (self._epoch - min_epoch), 2)
        return cur

    def _set_src_iter(self):
        min_epoch, max_epoch, min_iter, max_iter = self._src_iter_schedule
        if self._epoch <= min_epoch:
            y = min_iter
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_iter - min_iter) + min_iter
        self._src_iters = int(y)

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space
        _, _, min_len, max_len = self._rollout_schedule
        if self._src_epoch_stop < self._epoch and self._src_reset_rollout_length > 0:
            rlength = max_len
        else:
            rlength = self._rollout_length
        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq  # 100e3 * 1e3 / 250
        model_steps_per_epoch = int(rlength * rollouts_per_epoch)  # vary_number * cur_rollouts
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ Allocate Model Pool ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ Reallocate Model Pool ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _reallocate_use_src_model_pool(self):
        if hasattr(self, '_use_src_model_pool'):
            del self._use_src_model_pool
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space
        _, _, min_len, max_len = self._rollout_schedule
        if self._src_epoch_stop < self._epoch and self._src_reset_rollout_length > 0:
            rlength = max_len
        else:
            rlength = self._rollout_length
        new_pool_size = int(self._src_pool_size * rlength)
        new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
        self._use_src_model_pool = new_pool


    def _reallocate_use_src_model_pool_2(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space
        if self._src_epoch_stop < self._epoch and self._src_reset_rollout_length > 0:
            rlength = max_len
        else:
            rlength = self._rollout_length
        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq  # 100e3 * 1e3 / 250
        model_steps_per_epoch = int(rlength * rollouts_per_epoch)  # vary_number * cur_rollouts
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_use_src_model_pool'):
            print('[ Allocate Model Pool ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._use_src_model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
        elif self._use_src_model_pool._max_size != new_pool_size:
            print('[ Reallocate Src Model Pool ] Updating src model pool | {:.2e} --> {:.2e}'.format(
                self._use_src_model_pool._max_size, new_pool_size
            ))
            samples = self._use_src_model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._use_src_model_pool.size == new_pool.size
            self._use_src_model_pool = new_pool

    def _train_model(self, **kwargs):
        env_samples = self._pool.return_all_samples()
        train_inputs, train_outputs = format_samples_for_training(env_samples)
        model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _adapt_model(self, batch_size=256, max_steps=200, n_itr_critic=5):
        source_samples = self._pool.return_all_samples()
        target_samples = self._model_pool.return_all_samples()
        source_inputs, source_outputs = format_samples_for_training(source_samples)
        target_inputs, target_outputs = format_samples_for_training(target_samples)
        self._model.adapt(source_inputs, target_inputs, batch_size, max_steps=max_steps, n_itr_critic=n_itr_critic)

    def _train_model_by_src_function(self, **kwargs):
        # Iterate once for training

        if self._src_fix_iter:
            _, _, _, self._src_iters = self._src_iter_schedule
        # Split into training and holdout sets
        holdout_ratio = 0
        source_samples = self._pool.return_all_samples()
        true_sample_size = self._pool.size
        source_inputs, source_outputs = format_samples_for_training(source_samples)
        num_holdout = int(source_inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(source_inputs.shape[0])
        source_inputs, holdout_source_inputs = source_inputs[permutation[num_holdout:]], source_inputs[permutation[:num_holdout]]
        source_outputs, holdout_source_outputs = source_outputs[permutation[num_holdout:]], source_outputs[permutation[:num_holdout]]
        target_idx = permutation[:num_holdout]
        # before_src_samples_loss = self._src_fake_world_samples(target_idx)

        src_rollout_batch_size = int(self._src_pool_size)
        self._rollout_by_src_function(rollout_batch_size=int(src_rollout_batch_size),
                                      deterministic=self._deterministic)

        src_samples = self._use_src_model_pool.return_all_samples()
        train_inputs, train_outputs = format_samples_for_training(src_samples)
        aux = src_rollout_batch_size - true_sample_size
        num_hold = aux if aux > self._src_min_fake_sample_size else self._src_min_fake_sample_size
        if self._use_src_model_pool.size < self._src_min_fake_sample_size:
            num_hold = self._use_src_model_pool.size
        num_hold = int(num_hold)
        permutation = np.random.permutation(train_inputs.shape[0])
        # print("num_hold:",num_hold)
        train_inputs = train_inputs[permutation[:num_hold]]
        train_outputs = train_outputs[permutation[:num_hold]]
        train_inputs = np.concatenate((train_inputs, source_inputs), axis=0)
        train_outputs = np.concatenate((train_outputs, source_outputs), axis=0)
        model_metrics = self._model.use_src_train(train_inputs, train_outputs, **kwargs)
        del self._use_src_model_pool
        return model_metrics

    def _train_model_by_src_function_next_x_n_random(self, **kwargs):
        ##
        # Iterate n times for training
        # ##

        model_metrics = {}
        if self._src_fix_iter:
            _, _, _, self._src_iters = self._src_iter_schedule

        f_log = open(self.log_file, 'a')
        is_find_better = False

        # Split into training and holdout sets
        holdout_ratio = 0.69
        source_samples = self._pool.return_all_samples()
        source_inputs, source_outputs = format_samples_for_training(source_samples)
        num_holdout = int(source_inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(source_inputs.shape[0])
        source_inputs, holdout_source_inputs = source_inputs[permutation[num_holdout:]], source_inputs[permutation[:num_holdout]]
        source_outputs, holdout_source_outputs = source_outputs[permutation[num_holdout:]], source_outputs[permutation[:num_holdout]]
        target_idx = permutation[:num_holdout]
        # before_src_samples_loss = self._src_fake_world_samples(target_idx)
        f_log.write('begin src train \n')
        iters = 0
        for iters in range(self._src_iters):
            aux = int(self._rollout_batch_size / self._pool.size - 1)
            aux = 5 if aux > 5 else aux
            for i in range(aux):
                src_rollout_batch_size = int(self._pool.size)
                self._rollout_by_src_function(rollout_batch_size=int(src_rollout_batch_size),
                                              deterministic=self._deterministic)

            src_samples = self._use_src_model_pool.return_all_samples()
            train_inputs, train_outputs = format_samples_for_training(src_samples)
            num_holdout = int(self._pool.size * 5) if int(self._pool.size * 5) < train_inputs.shape[0] else train_inputs.shape[0]
            permutation = np.random.permutation(train_inputs.shape[0])
            train_inputs = train_inputs[permutation[:num_holdout]]
            train_outputs = train_outputs[permutation[:num_holdout]]
            train_inputs = np.concatenate((train_inputs, source_inputs), axis=0)
            train_outputs = np.concatenate((train_outputs, source_outputs), axis=0)
            model_metrics = self._model.use_src_train(train_inputs, train_outputs, **kwargs)

        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size  # vary_number, 100e3
        ))
        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        steps_added = []
        for i in range(self._rollout_length):
            act = self._policy.actions_np(obs)
            next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)

            steps_added.append(len(obs))

            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew,
                       'terminals': term}
            self._model_pool.add_samples(samples)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print(
                    '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length,
            self._n_train_repeat
        ))
        return rollout_stats

    def _rollout_by_src_function(self, rollout_batch_size, **kwargs):

        # Q_log = open("./Q_info.txt", 'a')
        batch = self.sampler.random_batch_truncation_trajectory(rollout_batch_size, k=self._rollout_length+1)
        # batch = self.sampler.order_batch_truncation_trajectory(rollout_batch_size, k=self._rollout_length)
        obs = batch['observations']
        next_obs = batch['next_observations']
        act = batch['actions']
        reward = batch['rewards']
        terminal = batch['terminals']
        steps_added = []
        cur_obs = obs[:, 0]
        real_obs = obs

        rollout_src_length = self._rollout_length
        if self._rollout_length <= 1:
            rollout_src_length = 2
        # Q_info = {}
        for j in range(rollout_src_length):
            real_cur_obs = real_obs[:, j]
            cur_act = act[:, j]
            cur_next_obs = next_obs[:, j]
            cur_rew = reward[:, j]
            cur_term = terminal[:, j]

            _next_obs, _rew, _term, _info = self.fake_env.step(cur_obs, cur_act, **kwargs)
            _next_obs = _next_obs.astype(np.float32)

            if j > 0:
                # compute Q
                bh = 2e7
                ins = math.ceil(len(real_cur_obs)/bh * 1.0)
                src_sample_added_number = 0

                for i in range(ins):
                    remain_set_range = int(len(real_cur_obs)%bh) if i==ins-1 else bh
                    start_index = int(i*bh)
                    end_index = int(i*bh+remain_set_range)
                    real_qs = self._get_Qs_critic_simulated(real_cur_obs[start_index: end_index], cur_act[start_index: end_index])
                    src_qs = self._get_Qs_critic_simulated(cur_obs[start_index: end_index], cur_act[start_index: end_index])

                    q_tau = self._reset_Q_tau()
                    q_samples_index = abs(real_qs[0] - src_qs[0]) <= q_tau
                    q_samples_index = q_samples_index.squeeze(-1)
                    samples = {'observations': cur_obs[start_index: end_index][q_samples_index], 'actions': cur_act[start_index: end_index][q_samples_index], 'next_observations': cur_next_obs[start_index: end_index][q_samples_index],
                               'rewards': cur_rew[start_index: end_index][q_samples_index],
                               'terminals': cur_term[start_index: end_index][q_samples_index]}

                    self._use_src_model_pool.add_samples(samples)

                    src_sample_added_number = src_sample_added_number + len(cur_obs[start_index: end_index][q_samples_index])
                steps_added.append(src_sample_added_number)
                # print("src_sample_added_number: ", src_sample_added_number)

            _nonterm_mask = ~_term.squeeze(-1)
            if _nonterm_mask.sum() == 0:
                print(
                    '[ Model Rollout ] Breaking early: {} | {} / {}'.format(j, _nonterm_mask.sum(),
                                                                            _nonterm_mask.shape))
                break
            nonterm_mask = ~cur_term.squeeze(-1)

            mask = np.logical_and(_nonterm_mask, nonterm_mask)
            real_obs = real_obs[mask]
            act = act[mask]
            next_obs = next_obs[mask]
            reward = reward[mask]
            terminal = terminal[mask]
            cur_obs = _next_obs[mask]
        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'by src function mean_rollout_length': mean_rollout_length}
        print(
            '[ By src function Model Rollout ] Added: {:.1e} | src Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
                sum(steps_added), self._use_src_model_pool.size, self._use_src_model_pool._max_size,
                mean_rollout_length,
                self._n_train_repeat
            ))
        return rollout_stats

    def _src_fake_world_samples(self, idx, **kwargs):
        samples = self._pool.return_all_samples()
        obs = samples['observations'][idx]
        # act = samples['actions']
        next_obs = samples['next_observations']
        terminals = samples['terminals']
        # rew = samples['rewards']
        loss_list = []
        for i in range(self._rollout_length):
            _act = self._policy.actions_np(obs)
            _next_obs, _rew, _term, _info = self.fake_env.step(obs, _act, **kwargs)
            loss_list = np.append(np.sum((next_obs[idx+i] - _next_obs) ** 2, axis=-1), loss_list)
            out_index = np.where((idx+i+1) >= len(next_obs))
            _term[out_index] = True
            _nonterm_mask = ~_term.squeeze(-1)
            nonterm_mask = ~terminals[idx+i].squeeze(-1)
            mask = np.logical_and(_nonterm_mask, nonterm_mask)
            obs = _next_obs[mask]
            idx = idx[mask]
        return np.mean(loss_list)

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size * self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        env_batch = self._pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)

            keys = env_batch.keys()
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def _training_batch_real(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        batch = self._pool.random_batch(batch_size)

        return batch

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._real_states = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation_real',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._real_next_states = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation_real',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._real_actions = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions_real',
        )

        self._real_next_actions = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions_next_real',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._real_rewards = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards_real',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        self._real_terminals = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals_real',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def  _get_Qs_critic_simulated(self, states, actions):

        """
            compute Q of self-correcting sumulated samples
            Returns Q
            -------
        """
        Q_values = tuple(Q([states, actions]) for Q in self._Q_critic_simulation)
        Q_values = self._session.run(Q_values)
        return Q_values


    def _init_critic_simulated_update(self):
        next_actions = self._policy.actions([self._real_next_states])
        Q_values = tuple(Q([self._real_next_states, next_actions]) for Q in self._Q_critic_simulation)
        b_value = Q_values[1]
        next_value = Q_values[0]

        Q_critic_simulated_target = td_target(
            reward=self._reward_scale * self._real_rewards,
            discount=self._discount,
            next_value=(1 - self._real_terminals) * next_value)

        assert Q_critic_simulated_target.shape.as_list() == [None, 1]

        Q_critic_simulated = tuple(
            Q([self._real_states, self._real_actions])
            for Q in self._Q_critic_simulation)

        Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_critic_simulated_target, predictions=Q_value, weights=1)
            for Q_value in Q_critic_simulated)

        self._Q_simulated_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_simulated_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Q_critic_simulation))

        Q_simulated_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                               "loss", "gradients", "gradient_norm", "global_gradient_norm"
                           ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Q_critic_simulation, Q_losses, self._Q_simulated_optimizers)))

        self._real_training_Q_ops.update({'Q_simulated': tf.group(Q_simulated_training_ops)})

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                               "loss", "gradients", "gradient_norm", "global_gradient_norm"
                           ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                    alpha * log_pis
                    - min_Q_log_target
                    - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            self._update_target()

    def _do_training_real_Q(self, batch):
        feed_dict = self._get_real_feed_dict(batch)
        self._session.run(self._real_training_Q_ops, feed_dict)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def _get_real_feed_dict(self, batch):
        feed_dict = {
            self._real_states: batch['observations'],
            self._real_actions: batch['actions'],
            self._real_next_states: batch['next_observations'],
            self._real_rewards: batch['rewards'],
            self._real_terminals: batch['terminals']
        }
        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables


    def save_diagnostics(self, diagnostics):
        csv_file = self.diagnostics_file

        previous_diagnostics = []
        try:
            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    previous_diagnostics.append(row)
        except FileNotFoundError:
            pass

        previous_diagnostics.append(diagnostics)

        with open(csv_file, 'w', newline='') as file:
            fieldnames = diagnostics.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(previous_diagnostics)

    def constructor_model_metrics(self, mesource, metraget):

        for key in mesource.keys():
            if key not in metraget.keys():
                metraget[key] = [mesource[key]]
            else:
                metraget[key].append(mesource[key])
        return metraget
