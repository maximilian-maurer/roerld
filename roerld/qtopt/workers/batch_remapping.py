def _remap_batch_keys(batch, prefix, include_action):
    new_entries = [(k[len(prefix):], v) for k, v in batch.items() if k.startswith(prefix)]
    if include_action:
        new_entries.append(("actions", batch["actions"]))
    return dict(new_entries)


def remap_keys_next_observations_in_batch_with_actions(batch):
    """Returns a new dictionary with the following properties:
        * the key "actions" is always in the new dictionary
        * all keys in batch that do not begin with next_observations_ are not in the new dictionary
        * all keys which do start with it are in the dictionary, but without that prefix
     """
    return _remap_batch_keys(batch, "next_observations_", True)


def remap_keys_observations_in_batch_with_actions(batch):
    """Returns a new dictionary with the following properties:
        * the key "action" is always in the new dictionary
        * all keys in batch that do not begin with next_observations_ are not in the new dictionary
        * all keys which do start with it are in the dictionary, but without that prefix
     """
    return _remap_batch_keys(batch, "observations_", True)


def remap_keys_next_observations_in_batch_without_actions(batch):
    """Returns a new dictionary with the following properties:
        * the key "actions" is always in the new dictionary
        * all keys in batch that do not begin with next_observations_ are not in the new dictionary
        * all keys which do start with it are in the dictionary, but without that prefix
     """
    return _remap_batch_keys(batch, "next_observations_", False)


def remap_keys_observations_in_batch_without_actions(batch):
    """Returns a new dictionary with the following properties:
        * the key "action" is always in the new dictionary
        * all keys in batch that do not begin with next_observations_ are not in the new dictionary
        * all keys which do start with it are in the dictionary, but without that prefix
     """
    return _remap_batch_keys(batch, "observations_", False)
