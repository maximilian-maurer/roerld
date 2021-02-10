import ray


class StaticRolloutManager:
    def __init__(self,
                 runner,
                 workers,
                 min_training_rollouts_per_epoch,
                 max_training_rollouts_per_epoch,

                 render_every_n_training_rollouts: int = 0):
        self.runner = runner
        self.min_training_rollouts_per_epoch = min_training_rollouts_per_epoch
        self.max_training_rollouts_per_epoch = max_training_rollouts_per_epoch

        assert self.min_training_rollouts_per_epoch <= self.max_training_rollouts_per_epoch

        self.rollout_actors = workers
        self.task_assignments = {}
        self.pending_eval_futures = []

        self.training_rollouts_enabled = False
        self.onpolicy_rollout_weights = None
        self.epoch_index = -1
        self._training_rollouts_started_this_epoch = 0
        self.round_robin_index = 0

        self.render_every_n_training_rollouts = render_every_n_training_rollouts
        self.total_training_rollouts = 0
        self.training_rollouts_arrived_this_epoch = 0

    def enable_training_rollouts(self):
        self.training_rollouts_enabled = True

    def _render_video_next_training_rollout(self):
        return self.render_every_n_training_rollouts > 0 and \
               self.total_training_rollouts % self.render_every_n_training_rollouts == 0

    def start_of_epoch(self, epoch_index, onpolicy_rollout_weights):
        self.onpolicy_rollout_weights = onpolicy_rollout_weights
        self.epoch_index = epoch_index
        self._training_rollouts_started_this_epoch = 0
        self.training_rollouts_arrived_this_epoch = 0

        if not self.training_rollouts_enabled:
            return

        # if there are free workers, start onpolicy-rollouts on them
        tasks_per_worker = {w: [t for t, iw in self.task_assignments.items() if iw == w] for w in self.rollout_actors}
        idle_workers = [w for w, t in tasks_per_worker.items() if len(t) == 0]

        futures_to_watch = []
        for worker in idle_workers:
            if self._training_rollouts_started_this_epoch > self.max_training_rollouts_per_epoch:
                break

            future = worker.rollout.remote(1,
                                           self.onpolicy_rollout_weights,
                                           {"epoch": self.epoch_index},
                                           False,
                                           self._render_video_next_training_rollout())
            self.task_assignments[future] = worker
            self._training_rollouts_started_this_epoch += 1
            futures_to_watch.append(future)
            self.total_training_rollouts += 1

        return futures_to_watch

    def schedule_evaluation_rollout(self, *args):
        # select a worker
        tasks_per_worker = {w: [t for t, iw in self.task_assignments.items() if iw == w] for w in self.rollout_actors}
        idle_workers = [w for w, t in tasks_per_worker.items() if len(t) == 0]
        evaluation_worker = None
        if len(idle_workers) > 0:
            evaluation_worker = idle_workers[0]
        else:
            evaluation_worker = self.rollout_actors[self.round_robin_index]
            self.round_robin_index = (self.round_robin_index + 1) % len(self.rollout_actors)

        rollout_future = evaluation_worker.rollout.remote(*args)
        self.pending_eval_futures.append(rollout_future)
        self.task_assignments[rollout_future] = evaluation_worker
        return [rollout_future]

    def manually_schedule_training_rollout(self, fully_random=False):
        # select a worker
        tasks_per_worker = {w: [t for t, iw in self.task_assignments.items() if iw == w] for w in self.rollout_actors}
        idle_workers = [w for w, t in tasks_per_worker.items() if len(t) == 0]
        evaluation_worker = None
        if len(idle_workers) > 0:
            evaluation_worker = idle_workers[0]
        else:
            evaluation_worker = self.rollout_actors[self.round_robin_index]
            self.round_robin_index = (self.round_robin_index + 1) % len(self.rollout_actors)

        new_future = evaluation_worker.rollout.remote(1,
                                                      self.onpolicy_rollout_weights,
                                                      {"epoch": self.epoch_index},
                                                      False,
                                                      self._render_video_next_training_rollout(),
                                                      fully_random=fully_random)
        self._training_rollouts_started_this_epoch += 1
        self.total_training_rollouts += 1
        self.task_assignments[new_future] = evaluation_worker

    def are_training_rollouts_enabled(self):
        return self.training_rollouts_enabled

    def future_arrived(self, future):
        if future in self.pending_eval_futures:
            self.runner.receive_evaluation_rollout(future)
            self.pending_eval_futures.remove(future)
        else:
            # print("Received training rollout")
            self.runner.receive_training_rollout(future)

        self.training_rollouts_arrived_this_epoch += 1
        #print(f"Training future arrived {self.training_rollouts_arrived_this_epoch}")

        if self.training_rollouts_enabled and \
                self._training_rollouts_started_this_epoch < self.max_training_rollouts_per_epoch:
            corresponding_actor = self.task_assignments[future]

            start_new_training_rollout = True
            if self._training_rollouts_started_this_epoch >= self.min_training_rollouts_per_epoch:
                # it is no longer absolutely imperative to schedule the rollout, so only do it if there is space for it
                this_worker_tasks = len([t for t, iw in self.task_assignments.items() if iw == corresponding_actor])
                if this_worker_tasks > 0:
                    start_new_training_rollout = False

            if start_new_training_rollout:
                # print("Started training rollout")
                new_future = corresponding_actor.rollout.remote(1,
                                                                self.onpolicy_rollout_weights,
                                                                {"epoch": self.epoch_index},
                                                                False,
                                                                self._render_video_next_training_rollout())
                self._training_rollouts_started_this_epoch += 1
                self.total_training_rollouts += 1
                self.task_assignments[new_future] = corresponding_actor

        del self.task_assignments[future]

    @property
    def training_rollouts_started_this_epoch(self):
        return self._training_rollouts_started_this_epoch

    def needs_stall(self):
        if self.training_rollouts_enabled and \
                self.training_rollouts_arrived_this_epoch < self.min_training_rollouts_per_epoch:
            return True
        if len(self.pending_eval_futures) <= len(self.rollout_actors):
            return False
        return True

    def pending_futures(self):
        return list(self.task_assignments.keys())

    def _is_pending_eval_future(self, future):
        return future in self.pending_eval_futures

    def close(self):
        futures = []
        for worker in self.rollout_actors:
            futures.append(worker.close.remote())
        ray.wait(futures, num_returns=len(futures))
