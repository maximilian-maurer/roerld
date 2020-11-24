
def rollout_experience_into_episodes(rollout_experience, episode_starts):
    """
    Converts the rollout experience (which is in the form of dictionary of one numpy array containing the results
    from one or more rollouts together with a list with info of where each episode is in the buffer) into a list of
    dictionaries containing the experience from a single episode each.
    """
    episode_starts_extended = [i for i in episode_starts]
    episode_starts_extended.append(len(rollout_experience["dones"]))

    episodes = []
    for i in range(len(episode_starts_extended) - 1):
        start = episode_starts_extended[i]
        end = episode_starts_extended[i + 1]

        this_episode = {
            k: rollout_experience[k][start:end] for k in rollout_experience.keys()
        }
        episodes.append(this_episode)

    return episodes
