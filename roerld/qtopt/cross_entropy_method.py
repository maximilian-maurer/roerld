from typing import Callable, Union

import numpy as np


def cross_entropy_method_normal(objective_function: Callable[[np.ndarray], np.ndarray],
                                initial_mean: Union[float, np.ndarray],
                                initial_std: Union[float, np.ndarray],
                                max_num_iterations: int,
                                sample_count: int,
                                elite_sample_count: int,
                                clip_samples=False,
                                clip_min=-1,
                                clip_max=-2):
    assert max_num_iterations > 0
    assert sample_count > 0
    assert elite_sample_count > 0
    assert elite_sample_count <= sample_count

    multi_dimensional = False
    if not np.isreal(initial_mean) is True:
        multi_dimensional = True
        assert len(initial_mean) == len(initial_std)

    if clip_samples:
        assert clip_min < clip_max

    mean = initial_mean
    std = initial_std
    samples = None
    evaluated_samples = None

    for i in range(max_num_iterations):
        if multi_dimensional:
            samples = mean + std * np.random.normal(loc=0, scale=1,
                                                    size=(sample_count, len(mean)))
        else:
            samples = mean + std * np.random.normal(loc=0, scale=1,
                                                    size=sample_count)
        if clip_samples:
            samples = np.clip(samples, clip_min, clip_max)

        evaluated_samples = objective_function(samples)
        partition_indices = np.argpartition(evaluated_samples, -elite_sample_count)[-elite_sample_count:]
        elite_samples = samples[partition_indices]
        mean = np.mean(elite_samples, axis=0)
        std = np.std(elite_samples, axis=0, ddof=1)

    best_sample = samples[np.argmax(evaluated_samples)]

    return best_sample


def cross_entropy_method_normal_batched(
        objective_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        objective_function_parameters: np.ndarray,
        num_items,
        initial_mean: np.ndarray,
        initial_std: np.ndarray,
        max_num_iterations: int,
        sample_count: int,
        elite_sample_count: int,
        clip_samples=False,
        clip_min=-1,
        clip_max=-2):
    """
    Runs multiple CEM optimizations as a batch (there is only one sample request for each cem iteration irrespective
     of how many parameter sets are given in objective_function_parameters).

    Compared to cross_entropy_method_normal the objective function is changed from:
        objective(samples) -> values
    To
        objective([params_1, params_2, ...], [samples_1, samples_2, ...])

    This function must run the objective_(params_i) on samples_i.

    Each samples_i is guaranteed to have the same length (sample_count), and there is exactly one sample request array
     for each array in params. The objective_function_parameters are passed unaltered to objective_function.
    objective_function is only called sequentially throughout, never parallel.
    """
    assert max_num_iterations > 0
    assert sample_count > 0
    assert elite_sample_count > 0
    assert elite_sample_count <= sample_count
    assert len(initial_mean) == len(initial_std)
    assert num_items > 0

    # todo this part of the api is deprecated
    assert not clip_samples

    if clip_samples:
        assert clip_min < clip_max

    mean = np.repeat([initial_mean], num_items, axis=0)
    std = np.repeat([initial_std], num_items, axis=0)
    all_evaluated_samples = None

    for _ in range(max_num_iterations):
        sample_requests = np.random.normal(loc=0, scale=1, size=(num_items, sample_count, len(initial_mean)))
        for of_param_index in range(num_items):
            sample_requests[of_param_index] = mean[of_param_index] + std[of_param_index] * sample_requests[of_param_index]

        all_evaluated_samples = objective_function(objective_function_parameters, sample_requests)

        partition_indices = np.argpartition(all_evaluated_samples, -elite_sample_count, axis=1)[:, -elite_sample_count:]
        all_elite_samples = sample_requests[np.arange(len(all_evaluated_samples))[:, None], partition_indices]

        mean = np.mean(all_elite_samples, axis=1)
        std = np.std(all_elite_samples, axis=1, ddof=1)

    best_samples = sample_requests[np.arange(len(sample_requests)), np.argmax(all_evaluated_samples, axis=1)]

    return best_samples



def cross_entropy_method_normal_batched(
        objective_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        objective_function_parameters: np.ndarray,
        num_items,
        initial_mean: np.ndarray,
        initial_std: np.ndarray,
        max_num_iterations: int,
        sample_count: int,
        elite_sample_count: int,
        clip_samples=False,
        clip_min=-1,
        clip_max=-2):
    """
    Runs multiple CEM optimizations as a batch (there is only one sample request for each cem iteration irrespective
     of how many parameter sets are given in objective_function_parameters).

    Compared to cross_entropy_method_normal the objective function is changed from:
        objective(samples) -> values
    To
        objective([params_1, params_2, ...], [samples_1, samples_2, ...])

    This function must run the objective_(params_i) on samples_i.

    Each samples_i is guaranteed to have the same length (sample_count), and there is exactly one sample request array
     for each array in params. The objective_function_parameters are passed unaltered to objective_function.
    objective_function is only called sequentially throughout, never parallel.
    """
    assert max_num_iterations > 0
    assert sample_count > 0
    assert elite_sample_count > 0
    assert elite_sample_count <= sample_count
    assert len(initial_mean) == len(initial_std)
    assert num_items > 0

    # todo this part of the api is deprecated
    assert not clip_samples

    if clip_samples:
        assert clip_min < clip_max

    mean = np.repeat([initial_mean], num_items, axis=0)
    std = np.repeat([initial_std], num_items, axis=0)
    all_evaluated_samples = None

    for _ in range(max_num_iterations):
        sample_requests = np.random.normal(loc=0, scale=1, size=(num_items, sample_count, len(initial_mean)))
        for of_param_index in range(num_items):
            sample_requests[of_param_index] = mean[of_param_index] + std[of_param_index] * sample_requests[of_param_index]

        all_evaluated_samples = objective_function(objective_function_parameters, sample_requests)

        partition_indices = np.argpartition(all_evaluated_samples, -elite_sample_count, axis=1)[:, -elite_sample_count:]
        all_elite_samples = sample_requests[np.arange(len(all_evaluated_samples))[:, None], partition_indices]

        mean = np.mean(all_elite_samples, axis=1)
        std = np.std(all_elite_samples, axis=1, ddof=1)

    best_samples = sample_requests[np.arange(len(sample_requests)), np.argmax(all_evaluated_samples, axis=1)]

    return best_samples

