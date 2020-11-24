import numpy as np
from roerld.qtopt.cross_entropy_method import cross_entropy_method_normal, cross_entropy_method_normal_batched


def test_ensure_parallel_close_to_single():
    objectives = [lambda xs: [-(x[0] ** 2) for x in xs],
                  lambda xs: [-(x[0] ** 2 - x[0]) for x in xs],
                  lambda xs: [-(x[0] ** 2 + x[0]) for x in xs],
                  lambda xs: [-((x[0] - 2) ** 2) for x in xs]]

    # compute results from single optimization
    results_single_runs = []
    for i in range(len(objectives)):
        results_single_runs.append(cross_entropy_method_normal(
            objectives[i],
            np.zeros(shape=1),
            np.ones(shape=1),
            5,
            100,
            10,
            False
        ))

    results_single_runs = np.array(results_single_runs)

    def parallel_objective(params, samples):
        # params is n index into objectives for each of the sample arrays in this case
        res = []
        for obj_idx, samples in zip(params, samples):
            res.append(objectives[obj_idx](samples))
        return np.array(res)

    results_parallel_run = cross_entropy_method_normal_batched(
        parallel_objective,
        np.array(list(range(len(objectives)))),
        len(objectives),
        np.zeros(shape=1),
        np.ones(shape=1),
        5,
        100,
        10,
        False
    )

    np.testing.assert_allclose(results_parallel_run, results_single_runs, rtol=0.05, atol=0.0001)

