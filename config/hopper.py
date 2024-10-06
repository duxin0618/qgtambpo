params = {
    'type': 'QGTAMBPO',
    'universe': 'gym',
    'domain': 'Hopper',
    'task': 'v2',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -1,
        'max_model_t': None,

        'rollout_schedule': [1, 45, 2, 20],

        'po_stop_epoch': 10000,

        'use_src': True,
        'src_iter_schedule': [1, 100, 1, 1],
        'src_fix_iter': True,
        'src_max_length': 200,
        'src_min_length': 1,
        'src_pool_size': 50e3,
        'src_min_fake_sample_size': 10e3,
        'plan': 6,
        'src_epoch_stop': 30,
        'Q_tau_info': [1, 30, 0.05, 0.08],
        'src_reset_rollout_length': -1  # 0 is false,
    }
}
