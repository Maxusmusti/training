import torch.multiprocessing as mp
import time
import uuid
from tqdm import trange


class TinyTrainer:
    """
    Handles training loop for a given distributed process address space.
    """

    def __init__(self):
        self.id = uuid.uuid1()

    def train(self, local_rank: int, world_size: int, epochs: int):
        print(
            f"TinyTrainerID: {self.id}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}"
        )
        for i in trange(epochs):
            time.sleep(local_rank * 0.1)


class TrainHarness:
    """Wraps a Trainer class that's been configured and is ready to be launched.
    Spawns `n_proc` based on GPU availability. Each proc creates its own Trainer
    object and calls 'train' on that object.
        Trainer objects ought to expect parallel, distributed invocation.
    """

    def __init__(self, n_parallel: int):
        self.n_parallel = n_parallel
        self.world_size: int = 5

    def _iter(self, local_rank: int, world_size: int):
        """
        Will be called within the context of its object but in
        parallel with other invokations.
        """
        print(f"LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

        # each object should be its own instance.
        trainer = TinyTrainer()
        trainer.train(local_rank, world_size, epochs=10)

    def train(self):
        mp.spawn(self._iter, args=(self.world_size,), nprocs=self.n_parallel)


if __name__ == "__main__":
    th = TrainHarness(n_parallel=5)
    th.train()
