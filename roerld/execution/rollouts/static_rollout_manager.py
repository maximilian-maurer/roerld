from typing import Iterable

import ray
from collections import deque
import numpy as np
from roerld.execution.ratios.limiter import SDLimiter
from roerld.execution.utils.opportunistic_actor_pool import OpportunisticActorPool
from roerld.execution.utils.waiting import flatten_list_of_lists, wait_all


class StaticRolloutManager:
    def __init__(self,
                 runner,
                 training_runs_started_limiters: Iterable[SDLimiter],
                 training_runs_returned_limiters: Iterable[SDLimiter],
                 render_every_n_training_rollouts: int = 0,
                 ):
        """
        Static rollout manager with soft limits. Will schedule such that:
            * For lower run start limits: At least this may training runs are scheduled each epoch
            * For lower return limits: At least this many returns will be had this epoch.
        However, as it is not possible to predict what time any rollout will take (and also impossible to predict,
        as in case of hardware failure, individual rollouts may not come back for several hours), maximums are more
        limited:
            * For upper run start limits: No new training run will be scheduled once any of these outputs True.

        Outside calls to `manually_schedule_training_rollout` ignore these rules, but the rollout manager will
        try to compensate by reducing its own rollouts accordingly, when possible.

        :param runner:
        :param workers:
        :param training_runs_started_limiters:
        :param training_runs_returned_limiters:
        :param render_every_n_training_rollouts:
        """
        self.runner = runner
        self.training_runs_started_limiters = list(training_runs_started_limiters)
        self.training_runs_returned_limiters = list(training_runs_returned_limiters)

        #if any([l.is_upper_limit() for l in training_runs_started_limiters]) \
        #        and any([l.is_lower_limit() for l in training_runs_returned_limiters]):
        #    raise ValueError("Cannot have an upper limit for the training runs started in one epoch if there is also "
        #                     "a lower limit on the number of returns.")

        self.actor_pool = OpportunisticActorPool([])

        self.pending_eval_futures = []

        self.training_rollouts_enabled = False
        self._onpolicy_rollout_weights = None
        self._epoch_index = -1
        self._training_rollouts_started_this_epoch = 0
        self._training_rollouts_arrived_this_epoch = 0

        self._render_every_n_training_rollouts = render_every_n_training_rollouts

        self.total_training_rollouts = 0

    def add_actor(self, actor):
        self.actor_pool.add_actor(actor)

    def start_of_epoch(self, epoch_index, onpolicy_rollout_weights):
        """
        Starts the next epoch.
        :param epoch_index: the epoch index
        :param onpolicy_rollout_weights: the weights to use for onpolicy rollouts or None if the old ones should be used.
        """
        if onpolicy_rollout_weights is not None:
            self._onpolicy_rollout_weights = onpolicy_rollout_weights
        self._epoch_index = epoch_index
        self._training_rollouts_started_this_epoch = 0
        self._training_rollouts_arrived_this_epoch = 0

        if not self.training_rollouts_enabled:
            return

        for lm in self.training_runs_started_limiters:
            lm.next_step(epoch_index)
        for lm in self.training_runs_returned_limiters:
            lm.next_step(epoch_index)

        futures = self.actor_pool.map_to_idle_actors(
            self._start_training_rollout_function(),
            lambda _: not any([limiter.upper_limit_reached() for limiter in self.training_runs_started_limiters]))
        futures_to_watch = flatten_list_of_lists(futures)

        return futures_to_watch

    def schedule_evaluation_rollout(self, *args):
        eval_futures = self.actor_pool.map_required_task(self._start_evaluation_rollout_function(*args))
        return flatten_list_of_lists(eval_futures)

    def manually_schedule_training_rollout(self, fully_random=False):
        futures = self.actor_pool.map_required_task(self._start_training_rollout_function(fully_random))
        futures_to_watch = flatten_list_of_lists(futures)
        return futures_to_watch

    def are_training_rollouts_enabled(self):
        return self.training_rollouts_enabled

    def _should_start_a_new_training_rollout(self):
        must_start_another_run = False
        if any([not l.lower_limit_achieved() for l in self.training_runs_started_limiters]):
            must_start_another_run = True
        if not must_start_another_run:
            # do the total remaining futures suffice to fulfill the limits? if not, we need to start another one
            remaining_train_futures = len([f for f in self.actor_pool.all_pending_tasks()
                                           if f not in self.pending_eval_futures])
            if any([not l.would_reach_lower_limit(remaining_train_futures)
                    for l in self.training_runs_returned_limiters]):
                must_start_another_run = True

        may_start_another_run = True
        if any([l.upper_limit_reached() for l in self.training_runs_started_limiters]):
            may_start_another_run = False

        if may_start_another_run is False and must_start_another_run is True:
            print("Warning: ignoring inconsistent limiters. Upper limit forbids starting new run, lower requires "
                  "starting one. Please review whether the combination of limits given to the rollout manager "
                  "is consistent.")

        if must_start_another_run:
            return True
        if not may_start_another_run:
            return False
        return True

    def future_arrived(self, future):
        if future in self.pending_eval_futures:
            self.runner.receive_evaluation_rollout(future)
            self.pending_eval_futures.remove(future)
        else:
            # print("Received training rollout")
            self.runner.receive_training_rollout(future)
            for limiter in self.training_runs_returned_limiters:
                limiter.one_step_performed()

        self._training_rollouts_arrived_this_epoch += 1
        # print(f"Training future arrived {self.training_rollouts_arrived_this_epoch}")

        self.actor_pool.task_done(future)
        del future
        if not self.training_rollouts_enabled:
            return

        if self._should_start_a_new_training_rollout():
            self.actor_pool.map_required_task(self._start_training_rollout_function())

    @property
    def training_rollouts_started_this_epoch(self):
        return self._training_rollouts_started_this_epoch

    def needs_stall(self):
        if self.training_rollouts_enabled:
            # stall if we did not start enough rollouts
            if any([not l.lower_limit_achieved() for l in self.training_runs_started_limiters]):
                return True
            # stall if we did not have enough rollouts returning
            if any([not l.lower_limit_achieved() for l in self.training_runs_returned_limiters]):
                return True
        # stall if we're overloading the workers with eval futures
        if len(self.pending_eval_futures) > self.actor_pool.num_actors:
            return True
        return False

    def pending_futures(self):
        return self.actor_pool.all_pending_tasks()

    def _is_pending_eval_future(self, future):
        return future in self.pending_eval_futures

    def close(self):
        waited = wait_all(flatten_list_of_lists(self.actor_pool.map_once_to_each_actor(lambda w: [w.close.remote()])))
        # collect exceptions if there were any
        for w in waited:
            ray.get(w)

    def _render_video_next_training_rollout(self):
        return self._render_every_n_training_rollouts > 0 and \
               self.total_training_rollouts % self._render_every_n_training_rollouts == 0

    def enable_training_rollouts(self):
        self.training_rollouts_enabled = True

    def _start_training_rollout_function(self, fully_random=False):
        def _inner(worker):
            future = worker.rollout.remote(1,
                                           self._onpolicy_rollout_weights,
                                           {"epoch": self._epoch_index},
                                           False,
                                           self._render_video_next_training_rollout(),
                                           fully_random=fully_random)
            self._training_rollouts_started_this_epoch += 1
            self.total_training_rollouts += 1
            for start_limiter in self.training_runs_started_limiters:
                start_limiter.one_step_performed()
            return [future]

        return _inner

    def _start_evaluation_rollout_function(self, *args):
        def _inner(worker):
            future = worker.rollout.remote(*args)
            self.pending_eval_futures.append(future)
            return [future]

        return _inner
