import os
import torch
import torch.distributed
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
import argparse
from datetime import timedelta
import math
import os
import time
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler, MistralForCausalLM
from torch.distributed import (
    ReduceOp,
    all_reduce,
)
import yaml
from dataclasses import dataclass

import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from multipack_sampler import find_packing_max_batch_len_and_grad_accum
from token_dataset import setup_dataloader, setup_dataset
from tokenizer_utils import setup_tokenizer
from utils import (
    save_hf_format_ds,
    set_random_seed,
    setup_logger,
    convert_loss_to_reduce_sum,
)

# NOTE: JAMES KUNSTLE add to fix -- RuntimeError: cutlassF: no kernel found to launch!
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def distributed_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8889"

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    deepspeed.init_distributed(timeout=timedelta(minutes=360))
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    # NOTE: below is for pytorch-native; doesn't work for deepspeed
    # init_process_group(backend="nccl", rank=rank, world_size=world_size)


@dataclass
class TrainerArgs:
    """Required arguments for the Trainer class."""

    model_name_or_path: str
    data_path: str
    output_path: str
    learning_rate: float
    n_epochs: int
    train_batch_size: int
    max_batch_len: int
    n_warmup_steps: int
    samples_per_save: int


class DeepSpeedTrainer:
    def __init__(self, args: TrainerArgs) -> None:
        """
        Order of initialization:
        1. Download and/or load tokenizer for model.
        2. Initialize DataSet.
        3. Calculate MultiPack-optimization values for batches.
        4. Setup DataLoader (with MultipackDistributedBatchSampler)
        5. Download and/or load model with loss layer optimizations (reduce_sum).
        6. Initialize optimizer.
        7. Configure DeepSpeed distributed training settings.
        8. Wrap model with DeepSpeed initialization class.


        Design considerations:
        1. Convinced that data loading + processing should NOT be in this class.
            Should be in its own class. However, for the time being, tightly coupling
            as a linear pipeline is convenient.
        2. Generally prefer functional-programming-style instantiation methods. i.e.
            if a method (e.g. _setup_tokenizer) has a side-effect (setting self.tokenizer)
            it should return a tokenizer rather than setting self.tokenizer internally.
            This style is preferred for readability.
            HOWEVER passing args to instantiation methods can be relaxed for the sake of
            brevity.
        3. All of the instance values (self.data_path, e.g.) are also in the self.args object.
            Choosing to copy the value into an instance-member is based on brevity.
        4.
        """

        # From TrainerArgs ------------------------------------------
        self.args: TrainerArgs = args
        self.model_name_or_path: str = args.model_name_or_path
        self.data_path: str = args.data_path
        self.world_size = (
            world_size
            if world_size
            else torch.distributed.get_world_size()  # TODO: get this from appropriate config.
        )
        self.local_rank = int(
            os.environ["LOCAL_RANK"]
        )  # TODO: get this from appropriate config.
        self.learning_rate: float = args.learning_rate
        self.num_warmup_steps: int = args.n_warmup_steps
        self.num_epochs: int = args.n_epochs
        self.max_batch_len: int = args.max_batch_len
        self.train_batch_size: int = args.train_batch_size
        self.samples_per_save = (
            args.samples_per_save // self.train_batch_size
        ) * self.train_batch_size

        # Data loading and utils -------------------------------------
        self.tokenizer = self._setup_tokenizer()
        self.dataset = self._setup_dataset()
        self.packing_max_batch_len, self.grad_accum_steps = (
            self._calc_packing_grad_accum()
        )
        self.train_micro_batch_size_per_gpu: int = self._calc_micro_batch_per_gpu()
        self.data_loader = self._setup_data_loader()

        # Model, Optimizer, DS configuration -------------------------
        self.model = (
            self._prep_model()
        )  # model needs a little setup before deepspeed-ifying
        self.optimizer = self._setup_optimizer()
        self.df_config = self._configure_deepspeed()
        self.model = (
            self._wrap_model_with_deepspeed()
        )  # okay now, wrap model w/ deepspeed.

        # Training status -----------------------------------
        self.global_step: int = 1

    def _calc_micro_batch_per_gpu(self):
        return (
            self.train_batch_size
            // self.grad_accum_steps
            // torch.distributed.get_world_size()
        )

    def _calc_packing_grad_accum(self):
        # NOTE: we need these params for the ds config and for setting up the
        # data loader.
        return find_packing_max_batch_len_and_grad_accum(
            num_gpus=torch.distributed.get_world_size(),
            avg_sample_len=self.dataset.get_lengths().mean(),
            train_batch_size=self.train_batch_size,
            max_batch_len_per_gpu=self.max_batch_len,
        )

    def _setup_tokenizer(self):
        return setup_tokenizer(model_name_or_path=self.model_name_or_path)

    def _setup_dataset(self):
        return setup_dataset(data_path=self.data_path, mock=False)

    def _setup_data_loader(self):

        return setup_dataloader(
            dataset=self.dataset,
            pad_token_id=self.tokenizer.pad_token_id,
            num_workers=8,
            is_granite=False,
            max_batch_len=self.max_batch_len,
            packing_max_batch_len=self.packing_max_batch_len,
            seed=self.seed,
        )

    def _wrap_model_with_deepspeed(self):
        """Wraps model object with deepspeed initializer and sets learning
        rate schedule based on training parameters.

        Returns:
            nn.Module: model
        """
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_epochs * len(self.data_loader),
        )

        model, _, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=self.df_config,
            lr_scheduler=lr_scheduler,
        )
        return model

    def _setup_optimizer(self):
        """Instantiates the optimizer.

        Returns:
            deepspeed.adam.AdamW: optimizer
        """

        # NOTE do we want to parameterize the beta coefficients?
        # TODO: should swap this for DeepSpeedCPUAdam if we enable CPU offloading.
        optimizer = FusedAdam(
            params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95)
        )
        return optimizer

    def _prep_model(self):
        """Downloads model from HF or loads from cache, exiting if
        model isn't supported. Enables model features.

        Returns:
            nn.Module: model
        """

        # TODO: need to look at model config to early-out if not a supported model
        # rather than download model and check later.

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        assert model.__class__.__name__ in [
            "MistralForCausalLM",  # NOTE: only support Mistral right now.
            # "GPTMegatronForCausalLM",
            # "LlamaForCausalLM"
        ], f"Model class name: {model.__class__.__name__} is not supported."

        model = convert_loss_to_reduce_sum(model)
        model.gradient_checkpointing_enable()
        return model

    def _configure_deepspeed(self):
        """Sets up DeepSpeed config based on the parameters
        passed to the trainig class.

        Returns:
            dict: DeepSpeed configuration object
        """
        return {
            # NOTE: shouldn't need train_batch_size if we're specifying
            # grad_accum_steps and train_micro_batch size.
            #
            # "train_batch_size": self.train_micro_batch_size_per_gpu
            # * self.world_size
            # * self.grad_accum_steps,
            "gradient_accumulation_steps": self.grad_accum_steps,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_param": {"device": "none"},
                "offload_optimizer": {
                    "device": "none"
                },  # TODO: parameterize offloading
            },
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }

    def _log_if_rank_zero(
        self, start_time, loss, num_loss_counted_tokens, aggregated_values
    ):

        if self.local_rank != 0:
            return

        elapsed_time = time.time() - start_time
        overall_throughput = (
            self.train_micro_batch_size_per_gpu * world_size / elapsed_time
        )
        current_lr = self.model.lr_scheduler.get_last_lr()[0]
        cuda_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        cuda_malloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]

        print(
            f"throughput: {overall_throughput} "
            f"samples/s, lr: {current_lr}, "
            f"loss: {loss.item()} "
            f"cuda_mem_allocated: {cuda_mem_allocated} GB "
            f"cuda_malloc_retries: {cuda_malloc_retries} "
            f"num_loss_counted_tokens: {num_loss_counted_tokens} "
            f"batch_size: {aggregated_values[1]} "
            f"total loss: {aggregated_values[2]/num_loss_counted_tokens}"
        )

    def _run_batch(self, batch, epoch):
        start_time = time.time()
        aggregated_values = torch.zeros(3, dtype=torch.float32).to(self.local_rank)
        aggregated_values[0] = batch["num_loss_counted_tokens"]
        aggregated_values[1] = len(batch["input_ids"])

        # NOTE: commented this out for the time being until we support granite.
        # if not args.is_granite:
        #     for k in batch:
        #         batch[k] = batch[k].to(local_rank)

        # move data over to GPU
        for k in batch:
            batch[k] = batch[k].to(self.local_rank)

        output = self.model(**batch, use_cache=False)
        loss = output.loss
        aggregated_values[2] = loss.item()

        # reduce aggregated_values structure across participating GPUs, summing each position
        all_reduce(aggregated_values, op=ReduceOp.SUM)

        num_loss_counted_tokens = aggregated_values[0]
        # dividing by the total number of non-padding tokens and multiplying by the number
        # of GPUs so when deepspeed averages by world_size, it will be the correct loss.
        loss = loss / num_loss_counted_tokens * world_size

        self.model.backward(loss)
        self.model.step()

        print(
            f"\033[93mPer-token loss scaled by world size: {(loss/num_loss_counted_tokens) * world_size}\033[0m"
        )
        print(
            f"Epoch: {epoch}, Step: {self.global_step}, Rank: {torch.distributed.get_rank()}, Loss = {loss}"
        )

        self._log_if_rank_zero(
            start_time=start_time,
            loss=loss,
            num_loss_counted_tokens=num_loss_counted_tokens,
            aggregated_values=aggregated_values,
        )

    def _run_epoch(self, epoch: int):
        self.data_loader.batch_sampler.set_epoch(epoch)

        if self.local_rank == 0:
            prog_bar = tqdm(range(len(self.data_loader)), desc=f"Epoch {epoch}")

        for batch in self.data_loader:

            # run a single forward and backward pass.
            self._run_batch(batch, epoch)

            if (
                self.global_step * self.train_micro_batch_size_per_gpu
            ) % self.samples_per_save == 0:
                self._save_checkpoint()

            self.global_step += 1
            if self.local_rank == 0:
                prog_bar.update(1)

            torch.cuda.empty_cache()

    def _save_checkpoint(self, epoch: int):
        # raise NotImplementedError()
        print("FAKE SAVING CHECKPOINT")
        return
        save_hf_format_ds(
            args,
            model,
            tokenizer,
            global_step * args.train_micro_batch_size_per_gpu * world_size,
        )

    def train(self):

        # sets model layers to "requires_grad", etc.
        self.model.train()

        self.global_step = 1
        (
            print(f"\033[93mNumber of samples per save: {self.samples_per_save}\033[0m")
            if self.local_rank == 0
            else None
        )

        for epoch in range(self.num_epochs):
            torch.distributed.barrier()
            self._run_epoch(epoch)


# class TrainHarness:

#     def __init__(self, n_parallel: int, trainer: Trainer):
#         self.n_parallel = n_parallel
#         self.world_size: int = 5

#     def _iter(self, local_rank: int, world_size: int):
#         """
#         Will be called within the context of its object but in
#         parallel with other invokations.
#         """
#         print(f"LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

#         # each object should be its own instance.
#         trainer = trainer()
#         trainer.train(local_rank, world_size, epochs=10)

#     def train(self):
#         mp.spawn(self._iter, args=(self.world_size,), nprocs=self.n_parallel)


def main(rank: int, world_size):
    # ======================================================
    # TODO: these are fake, move to proper config.
    train_args = TrainerArgs(
        model_name_or_path="./model",
        data_path="/dev/shm/train.jsonl",
        output_dir="./out",
        num_epochs=5,
        train_batch_size=32,
        learning_rate=1e-6,
        num_warmup_steps=385,
        local_rank=rank,
        world_size=world_size,
        seed=42,
        sharding_strat="HYBRID_SHARD",
        max_batch_len=200,
    )
    # ======================================================

    distributed_setup()
    trainer = DeepSpeedTrainer(args=train_args)
    trainer.train()
    torch.distributed.barrier()
    destroy_process_group()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str)
    # parser.add_argument("--data_path", type=str)

    # # where we'll store checkpoints.
    # parser.add_argument("--output_dir", type=str)
    # parser.add_argument("--num_epochs", type=int, default=1)

    # # (DeepSpeed doc definitions)
    # # the amount of data samples that leads to one step of model update.
    # # train_batch_size = train_micro_batch_size_per_gpu * grad_accum_steps * num_gpus
    # parser.add_argument("--train_batch_size", type=int, default=3840)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)

    # # Number of very-low-learning-rate steps that the model training flow takes.
    # # Hypothetically, helps the attention mechanism slowly acclimate to the data.
    # # Practically, helps the adaptive optimizer (Adam) compute good statistics on
    # # the gradients it's seeing.
    # parser.add_argument("--num_warmup_steps", type=int, default=1000)

    # # number of data points to be seen before the trainer saves a checkpoint
    # # of the model and training parameters.
    # parser.add_argument("--save_samples", type=int)
    # parser.add_argument("--log_level", type=str, default="INFO")
    # parser.add_argument("--seed", type=int, default=42)

    # # removing this for the time being.
    # # parser.add_argument("--mock_data", action="store_true")
    # # parser.add_argument("--mock_len", type=int, default=2600)
    # parser.add_argument(
    #     "--sharding_strategy",
    #     type=str,
    #     default="FULL_SHARD",
    #     help="Sharding strategy to be used for distributed training.",
    # )
    # parser.add_argument("--max_batch_len", type=int, default=60000)
    # args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size), nprocs=world_size)

"""
What we're using to run working scripts.

torchrun \
--nnodes=1 \
--nproc_per_node=gpu \
--node_rank=0 \
--standalone \
main_ds.py \
--model_name_or_path="/ilab-data/model" \
--data_path="/dev/shm/data.jsonl" \
--output_dir="out" \
--num_epochs=5 \
--train_batch_size=32 \
--learning_rate="1e-6" \
--num_warmup_steps=800 \
--save_samples=12000 \
--log_level="INFO" \
--seed=42 \

------------------------------------------------------------------

The parser that the existing script uses.

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--num_epochs", type=int, default=1)
# parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=8)
parser.add_argument("--train_batch_size", type=int, default=3840)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_warmup_steps", type=int, default=1000)
# parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--save_samples", type=int)
parser.add_argument("--log_level", type=str, default="INFO")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mock_data", action="store_true")
parser.add_argument("--mock_len", type=int, default=2600)
parser.add_argument(
    "--sharding_strategy",
    type=str,
    # choices=[e.name for e in ShardingStrategy],
    default="FULL_SHARD",
    help="Sharding strategy to be used for distributed training.",
)
parser.add_argument("--is_granite", action="store_true")
parser.add_argument("--max_batch_len", type=int, default=60000)
args = parser.parse_args()
"""
