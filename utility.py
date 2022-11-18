import random

from ray.rllib.utils import override
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.utils.typing import SampleBatchType


class RetentionReplayBufferRay(ReplayBuffer):
    """This buffer saves the initial r% of the buffer during eviction to prevent catastrophic forgetting"""

    def __init__(
            self,
            capacity: int = 10000,
            storage_unit: str = "timesteps",
            retention_rate: float = 0.1,
            **kwargs
    ):
        """Initializes a RetentionReplayBuffer instance.

        Args:
            capacity: Max number of timesteps to store in the FIFO
                buffer. After reaching this number, older samples will be
                dropped to make space for new ones.
            storage_unit: Either 'timesteps', 'sequences' or
                'episodes'. Specifies how experiences are stored.
            retention_rate: How much of the initial buffer to keep
                during eviction (r).
            ``**kwargs``: Forward compatibility kwargs.
        """
        ReplayBuffer.__init__(self, capacity, storage_unit, **kwargs)
        assert retention_rate > 0
        self.retention_rate = retention_rate

    @override(ReplayBuffer)
    def _add_single_batch(self, item: SampleBatchType, **kwargs) -> None:
        self._num_timesteps_added += item.count
        self._num_timesteps_added_wrap += item.count

        if self._next_idx >= len(self._storage):
            self._storage.append(item)
            self._est_size_bytes += item.size_bytes()
        else:
            item_to_be_removed = self._storage[self._next_idx]
            self._est_size_bytes -= item_to_be_removed.size_bytes()
            self._storage[self._next_idx] = item
            self._est_size_bytes += item.size_bytes()

        # Eviction of older samples has already started (buffer is "full").
        if self._eviction_started:
            self._evicted_hit_stats.push(self._hit_count[self._next_idx])
            self._hit_count[self._next_idx] = 0

        # Wrap around storage as a circular buffer once we hit capacity.
        if self._num_timesteps_added_wrap >= self.capacity:
            self._eviction_started = True
            self._num_timesteps_added_wrap = 0

            self._next_idx = 0 + int(self.capacity * self.retention_rate)  # keep r% of the buffer
            # self.retention_rate *= 0.5
        else:
            self._next_idx += 1
