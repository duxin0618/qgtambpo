from copy import deepcopy

import qgtambpo


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_QGTAMBPO_algorithm(variant, *args, **kwargs):

    algorithm = qgtambpo.QGTAMBPO(*args, **kwargs)

    return algorithm


# def create_AMPO_MMD_algorithm(variant, *args, **kwargs):
#     import ampo_mmd
#
#     algorithm = ampo_mmd.AMPO_MMD(*args, **kwargs)
#
#     return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'QGTAMBPO': create_QGTAMBPO_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
