import ray


def wait_all(futures):
    return ray.wait(futures, num_returns=len(futures))[0]


def flatten_list_of_lists(list_of_lists):
    result = []
    for l in list_of_lists:
        result.extend(l)
    return result
