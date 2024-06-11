import os
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import deepspeed
import torch
import torch.distributed
import torch.multiprocessing as mp
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from dolomite_engine.hf_models.models import GPTDolomiteForCausalLM
from torch.distributed import ReduceOp, all_reduce, destroy_process_group
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler

from multipack_sampler import find_packing_max_batch_len_and_grad_accum
from token_dataset import setup_dataloader, setup_dataset
from tokenizer_utils import setup_tokenizer
from utils import (
    convert_loss_to_reduce_sum,
    save_hf_format_ds,
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
    ckpt_path: str
    is_granite: bool = True

    num_gpus: int
    max_seq_len: int
    num_epochs: int
    effective_batch_size: int
    save_samples: int
    learning_rate: float
    warmup_steps: int

    ds_offload_strat: Enum["cpu", "nvme", None]
    cpu_offload_optimizer: bool
    cpu_offload_params: dict
    sharding_strat: Enum["HYBRID", "FULL"]

    quantize_dtype: Enum["nf4", "fp8", None]
    lora: bool
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: list

    seed: int = None
    world_size: int


class DataWrangler:
    """Collates dataset preparation steps
    into a single class, calculating required
    training parameters that the trainer can use.

    Name is temporary.
    """

    def __init__(self, args: TrainerArgs) -> None:
        self.args = args
        self.model_name_or_path = args.model_name_or_path
        self.data_path: str = args.data_path
        self.max_batch_len: int = args.max_batch_len
        self.train_batch_size: int = args.effective_batch_size
        self.samples_per_save = (
            args.save_samples // self.train_batch_size
        ) * self.train_batch_size

        # Data loading and utils -------------------------------------
        self.tokenizer = self._setup_tokenizer()
        self.dataset = self._setup_dataset()
        self.packing_max_batch_len, self.grad_accum_steps = (
            self._calc_packing_grad_accum()
        )
        self.train_micro_batch_size_per_gpu: int = self._calc_micro_batch_per_gpu()
        self.data_loader = self._setup_data_loader()

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

        # NOTE: mock is False by default for the moment, if we need
        # it we can parameterize it.
        return setup_dataset(data_path=self.data_path, mock=False)

    def _setup_data_loader(self):

        return setup_dataloader(
            dataset=self.dataset,
            pad_token_id=self.tokenizer.pad_token_id,
            num_workers=8,
            is_granite=False,
            max_batch_len=self.max_batch_len,
            packing_max_batch_len=self.packing_max_batch_len,
            seed=self.args.seed,
        )


class DeepSpeedTrainer:
    def __init__(self, args: TrainerArgs, data_wrangler: DataWrangler) -> None:
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
        1. Generally prefer functional-programming-style instantiation methods. i.e.
            if a method (e.g. _setup_tokenizer) has a side-effect (setting self.tokenizer)
            it should return a tokenizer rather than setting self.tokenizer internally.
            This style is preferred for readability.
            HOWEVER passing args to instantiation methods can be relaxed for the sake of
            brevity.
        2. All of the instance values (self.data_path, e.g.) are also in the self.args object.
            Choosing to copy the value into an instance-member is based on brevity.
        """

        # From TrainerArgs ------------------------------------------
        self.args: TrainerArgs = args
        self.model_name_or_path: str = args.model_name_or_path
        self.world_size = (
            args.world_size
            if args.world_size > 0
            else torch.distributed.get_world_size()  # TODO: get this from appropriate config.
        )
        self.local_rank = int(
            os.environ["LOCAL_RANK"]
        )  # TODO: get this from appropriate config.
        self.learning_rate: float = args.learning_rate
        self.num_warmup_steps: int = args.warmup_steps
        self.num_epochs: int = args.num_epochs

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
        self.data_loader = data_wrangler.data_loader
        self.data_wrangler = data_wrangler

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

        if self.args.is_granite:
            model = GPTDolomiteForCausalLM.from_pretrained(
                self.model_name_or_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                use_padding_free_transformer=True,
            )
        else:
            bnb_config = None

            # NOTE: James, updated from original because of config design doc.
            if self.args.lora_rank > 0 and self.args.quantize_dtype == "nf4":
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # TODO: parameterize for different data types.
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
                )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name_or_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )

        assert model.__class__.__name__ in [
            "MistralForCausalLM",
            "GPTDolomiteForCausalLM",
            "LlamaForCausalLM",
            "Starcoder2ForCausalLM",
            "GemmaForCausalLM",
        ], f"Model class name: {model.__class__.__name__} is not supported."

        model = convert_loss_to_reduce_sum(model)

        if self.args.is_granite:
            from dolomite_engine.enums import GradientCheckpointingMethod
            from dolomite_engine.gradient_checkpointing import (
                apply_gradient_checkpointing,
            )

            block_name = model._no_split_modules[0]
            apply_gradient_checkpointing(
                model,
                GradientCheckpointingMethod.block,
                block_name=block_name,
                use_reentrant=True,  # this should be the HF default mode
            )
        elif self.args.lora_r > 0:
            # if lora
            from peft import LoraConfig

            from utils import patch_target_module, prepare_peft_model

            if self.args.lora_target_modules is None:
                self.args.__dict__["target_modules"] = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]

            peft_config = LoraConfig(
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                r=self.args.lora_rank,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.args.target_modules,
            )
            prepare_peft_model(model, peft_config)

            # patch DS to work with quantized models
            from functools import partial

            from deepspeed import DeepSpeedEngine

            if self.args.lora_quant_bits is not None:
                patch_target_module(
                    "deepspeed.DeepSpeedEngine",
                    partial(DeepSpeedEngine, dont_change_device=True),
                )
        else:
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
            "gradient_accumulation_steps": self.data_loader.grad_accum_steps,
            "train_micro_batch_size_per_gpu": self.data_loader.train_micro_batch_size_per_gpu,
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
            self.data_loader.train_micro_batch_size_per_gpu
            * self.world_size
            / elapsed_time
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
        aggregated_values[0] = batch.pop("num_loss_counted_tokens")
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
        loss = loss / num_loss_counted_tokens * self.args.world_size

        self.model.backward(loss)
        self.model.step()

        print(
            f"\033[93mPer-token loss scaled by world size: {(loss/num_loss_counted_tokens) * self.args.world_size}\033[0m"
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

        for batch in self.data_loader.data_loader:

            # run a single forward and backward pass.
            self._run_batch(batch, epoch)

            if (
                self.global_step * self.data_wrangler.train_micro_batch_size_per_gpu
            ) % self.data_wrangler.samples_per_save == 0:
                self._save_checkpoint(epoch=epoch)

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

        if self.local_rank == 0:
            print(
                f"\033[93mNumber of samples per save: {self.data_wrangler.samples_per_save}\033[0m"
            )

        for epoch in range(self.num_epochs):
            torch.distributed.barrier()
            self._run_epoch(epoch)


def main():
    # ======================================================
    # NOTE: these are fake.
    train_args = TrainerArgs(
        model_name_or_path="./model",
        data_path="/dev/shm/train.jsonl",
        ckpt_path="./out",
        num_epochs=5,
        effective_batch_size=32,
        learning_rate=1e-6,
        num_warmup_steps=385,
        world_size=os.environ["WORLD_SIZE"],
        sharding_strat="HYBRID_SHARD",
    )
    # ======================================================

    distributed_setup()
    dataw = DataWrangler(train_args)
    trainer = DeepSpeedTrainer(args=train_args, data_wrangler=dataw)
    trainer.train()
    torch.distributed.barrier()
    destroy_process_group()


if __name__ == "__main__":

    # make sure this stuff is set.
    assert os.environ["WORLD_SIZE"] > 0
    assert os.environ["RANK"] >= 0

    main()
