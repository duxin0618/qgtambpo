params = {
    'type': 'QGTAMBPO',
    'universe': 'gym',
    'domain': 'Swimmer',
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

        'rollout_schedule': [1, 20, 2, 2],
        'po_stop_epoch': 20,

        'use_src': True,
        'src_iter_schedule': [1, 20, 1, 1],
        'src_fix_iter': True,
        'src_max_length': 10,
        'src_min_length': 1,
        'src_pool_size': 10e3,
        'src_min_fake_sample_size': 50e2,
        'plan': 6,
        'src_epoch_stop': 6,
    }
}

