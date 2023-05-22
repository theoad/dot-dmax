import warnings
from contextlib import ContextDecorator
from typing import Callable, List, Optional

import torch.distributed
from torch import Tensor


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def gather_all_if_ddp_available(tensors: Tensor) -> List[Tensor]:
    if is_distributed():
        if tensors.ndim == 0:
            tensors = tensors.clone()[None]
        output_tensors = [tensors.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensors)
        return output_tensors
    return [tensors]


def reduce_all_if_ddp_available(tensors: Tensor) -> Tensor:
    if is_distributed():
        return torch.distributed.all_reduce(tensors)
    return tensors


def rank_zero_print(msg: str) -> None:
    if not is_distributed() or torch.distributed.get_rank() == 0:
        warnings.warn(msg)


class rank_zero_first(ContextDecorator):
    def __init__(self):
        if is_distributed() and torch.distributed.get_rank():
            torch.distributed.barrier()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if is_distributed() and torch.distributed.get_rank() == 0:
            torch.distributed.barrier()
        return False


DDPAllGather = Callable[[Tensor], List[Tensor]]
DDPAllReduce = Callable[[Tensor], Tensor]
DDPWarn = Callable[[str], None]


class DDPMixin(object):
    def __init__(
            self,
            ddp_reduce_func: Optional[DDPAllReduce] = reduce_all_if_ddp_available,
            ddp_gather_func: Optional[DDPAllGather] = gather_all_if_ddp_available,
            ddp_warn_func: Optional[DDPWarn] = rank_zero_print
    ):
        self.reduce = ddp_reduce_func or (lambda x: x)
        self.gather = ddp_gather_func or (lambda x: [x])
        self.warn = ddp_warn_func or warnings.warn


def ema(moving_avg, new, decay):
    if decay is None: return moving_avg + new
    return moving_avg * decay + new * (1 - decay)


if __name__ == '__main__':
    # rank_zero_only_test
    from accelerate import Accelerator
    from time import sleep
    import os

    ddp = Accelerator()
    print(f"[rank {ddp.local_process_index}]: init")
    ddp.wait_for_everyone()
    cache = "cache.txt"

    with rank_zero_first():
        print(f"[rank {torch.distributed.get_rank()}]: entered")
        if os.path.exists(cache):
            with open(cache, "r") as f:
                s = f.readline()
        else:
            sleep(10)  # some complex computation
            s = f"hello world from rank {torch.distributed.get_rank()}"
            with open(cache, "w") as f:
                f.write(s)

    print(f"[rank {torch.distributed.get_rank()}]: {s}")

    if os.path.exists(cache):
        os.remove(cache)
