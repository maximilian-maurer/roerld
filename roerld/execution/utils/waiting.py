import ray


def wait_all(futures):
    ray.wait(futures, num_returns=len(futures))
