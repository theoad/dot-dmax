from typing import Optional, Type, List, Literal, Dict
import functools

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
import torchmetrics.image as ti
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM

from rich.table import Table
from rich.console import Console
from rich import box, print


class FID(ti.FrechetInceptionDistance):
    def __init__(self, **kwargs):
        super().__init__(**{'reset_real_features': False, 'normalize': True, **kwargs})

    def update(self, preds: Optional[Tensor] = None, target: Optional[Tensor] = None):
        if preds is not None: super().update(preds, real=False)
        if target is not None: super().update(target, real=True)


class KID(ti.KernelInceptionDistance):
    def __init__(self, **kwargs):
        super().__init__(**{'reset_real_features': False, 'normalize': True, **kwargs})

    def update(self, preds: Optional[Tensor] = None, target: Optional[Tensor] = None):
        if preds is not None: super().update(preds, real=False)
        if target is not None: super().update(target, real=True)

    def compute(self):
        res = super(KID, self).compute()
        return res[0] * 1000, res[1] * 1000

class IS(ti.InceptionScore):
    def __init__(self, **kwargs):
        super().__init__(**{'normalize': True, **kwargs})

    def update(self, preds: Optional[Tensor] = None, target: Optional[Tensor] = None):
        return super().update(preds)


class LPIPS(ti.LearnedPerceptualImagePatchSimilarity):
    def __init__(self, **kwargs):
        super().__init__(**{'net_type': "vgg", 'normalize': True, **kwargs})

    def update(self, preds: Optional[Tensor] = None, target: Optional[Tensor] = None):
        super().update(preds, target)


def map_args(**map):
    def mapped(method):
        @functools.wraps(method)
        def wrapper(self, **kwargs):
            matched = {map[k]: v for k, v in kwargs.items() if k in map.keys()}
            return method(self, **matched)
        return wrapper
    return mapped


TRAIN_METRICS: List[Type[Metric]] = [PSNR, SSIM]

VAL_METRICS: List[Type[Metric]] = TRAIN_METRICS + [LPIPS, IS, FID]

def filtered_metric(
        from_cls: Type[Metric],
        estimate_key: str = "mmse",
        clean_key: str = "clean",
        **metric_kwargs
) -> Metric:
        class FilteredKwargs(from_cls):
            @map_args(**{estimate_key: "preds", clean_key: "target"})
            def _filter_kwargs(self, **kwargs):
                return super()._filter_kwargs(**kwargs)  # noqa

        return FilteredKwargs(**metric_kwargs)


def acronym(word: str, lower: bool = True) -> str:
    """
    gets word acronym:
        >>> acronym("FrechetInceptionDistance", lower=False)
        >>> 'FID'
    """
    if all(c.islower() for c in word): return word
    acronym_upper = ''.join(list(filter(lambda c: c.isupper(), word)))
    return acronym_upper.lower() if lower else acronym_upper


def metrics(estimate: str, clean: str = "clean", mode: Literal["train", "val"] = "val"):
    return MetricCollection({
        acronym(metric_cls.__name__): filtered_metric(metric_cls, estimate, clean)
        for metric_cls in (VAL_METRICS if mode == "val" else TRAIN_METRICS)
    }, prefix=estimate+'/')


def generate_metrics(key_list: List[str], clean_str: str = "clean", mode: Literal["train", "val"] = "train") -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict({
        k: metrics(k, clean_str, mode=mode) for k in key_list
    })


def print_results(metric_dict, out_file: Optional[str] = None):
    results = {signal: metric.compute() for signal, metric in metric_dict.items()}
    first_res = next(iter(results.values()))

    # Create a table object
    table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL)

    # Add columns for the dictionary index and the keys
    table.add_column("Signal", style="dim", justify="center")
    for metric_name in first_res.keys():
        table.add_column(acronym(metric_name.split("/", 1)[-1], lower=False), justify="center")

    # Add rows for each dictionary
    for signal, res in results.items():
        row_values = [signal]
        for metric_name in res.keys():
            if isinstance(res[metric_name], tuple):
                row_values.append(f"{res[metric_name][0].cpu().item():.2f}±{res[metric_name][1].cpu().item():.2f}")
            else:
                row_values.append(f"{res[metric_name].cpu().item():.2f}")
        table.add_row(*row_values)

    # Render the table in the console with colors
    if out_file is None:
        console = Console()
        console.print(table)
    else:
        with open(out_file, "w") as file:
            print(table, file=file)
