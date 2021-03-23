from typing import Any, Union, Callable, List


class OpportunisticActorPool:
    """
    Actor pool tracking task allocations with a focus on operations on idle workers (as opposed to the normal
    actor pool which is driven by tasks that have to be performed) for optional operations.
    """
    def __init__(self, actors):
        self._actors = actors
        self._actors_to_tasks = {a: [] for a in self._actors}
        self._round_robin_index = 0

    @property
    def num_actors(self):
        return len(self._actors)

    def idle_actors(self):
        return [w for w, t in self._actors_to_tasks.items() if len(t) == 0]

    def try_first_idle_actor(self) -> Union[None, Any]:
        idle = self.idle_actors()
        if len(idle) > 0:
            return idle[0]
        return None

    def has_idle_actor(self):
        return self.try_first_idle_actor() is not None

    def associate_task(self, actor, task):
        if actor not in self._actors_to_tasks:
            raise ValueError("Cannot associate a task with actor which is not managed by the pool")
        self._actors_to_tasks[actor].append(task)

    def associate_tasks(self, actor, tasks):
        if actor not in self._actors_to_tasks:
            raise ValueError("Cannot associate a task with actor which is not managed by the pool")
        for task in tasks:
            self._actors_to_tasks[actor].append(task)

    def task_done(self, task):
        actor = [w for w, t in self._actors_to_tasks.items() if task in t]
        if len(actor) == 0:
            raise ValueError("Tried to mark a task in the pool as done despite it not being tracked here.")
        if len(actor) > 1:
            raise ValueError("Task is represented in multiple actors.")

        self._actors_to_tasks[actor[0]].remove(task)

    def all_pending_tasks(self):
        tasks = []
        for t in self._actors_to_tasks.values():
            tasks.extend(t)
        return tasks

    def map_to_idle_actors(self, map_fn: Callable[[Any], List[Any]], continue_fn: Callable[[int], bool] = None):
        """
        Maps a callable to idle actors. If continue_fn is None, this calls map_fn once for each idle actor.
        If continue_fn is not None, then after each worker for which map_fn has been called, it will be called
        once. If it then returns False, no further work is assigned to the remaining idle actors, and this function
        returns. continue_fn will be called once at the beginning and if it returns False then, no work will be started.

        :param continue_fn: see method description.
        :param map_fn: Takes actor and returns the list of tasks to associate with the actor.
        :return a list of the list of tasks for each worker that have been created
        """
        if continue_fn is not None and not continue_fn(0):
            return []

        idlers = self.idle_actors()
        workers_so_far = 0
        all_tasks = []
        for idle in idlers:
            tasks = map_fn(idle)
            self.associate_tasks(idle, tasks)
            all_tasks.append(tasks)
            workers_so_far += 1
            if continue_fn is not None and not continue_fn(workers_so_far):
                break
        return all_tasks

    @staticmethod
    def count_limiter(limit):
        return lambda so_far: so_far < limit

    def map_ongoing_round_robin(self, map_fn: Callable[[Any], List[Any]], continue_fn: Callable[[int], bool]):
        """
        Same semantics as map_to_idle_actors except that here all workers are available, and will be iterated over in a
        (persistent accross calls) round robin fashion. Consequently continue_fn cannot be None here, as it must
        provide a limit for when to stop.
        """
        workers_so_far = 0
        all_tasks = [[] for _ in range(len(self._actors))]
        keep_going = True
        while keep_going is True:
            idle = self._actors[self._round_robin_index]
            tasks = map_fn(idle)
            self.associate_tasks(idle, tasks)
            all_tasks[self._round_robin_index].extend(tasks)
            workers_so_far += 1
            self._round_robin_index = (self._round_robin_index + 1) % len(self._actors)
            keep_going = continue_fn(workers_so_far)
        return all_tasks

    def map_once_to_each_actor(self, map_fn: Callable[[Any], List[Any]]):
        """
        Same semantics as map_to_idle_actors except that here all workers are available, and will be iterated over in a
        (persistent accross calls) round robin fashion. Consequently continue_fn cannot be None here, as it must
        provide a limit for when to stop.
        """
        all_tasks = [[] for _ in range(len(self._actors))]
        for idx, idle in enumerate(self._actors):
            tasks = map_fn(idle)
            self.associate_tasks(idle, tasks)
            all_tasks[idx].extend(tasks)
        return all_tasks

    def map_required_task(self, map_fn: Callable[[Any], List[Any]]):
        """
        Maps a non-optional task.
        """
        if self.has_idle_actor():
            return self.map_to_idle_actors(map_fn, self.count_limiter(1))
        return self.map_ongoing_round_robin(map_fn, self.count_limiter(1))

    def add_actor(self, actor):
        self._actors.append(actor)
        self._actors_to_tasks[actor] = []
