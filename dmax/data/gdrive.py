import os
from typing import Optional, List, Dict, Literal
import jsonargparse
import shutil
import json
from dmax.utils import CACHE
from dmax.utils.drive import build_service, download_file

with open(("dmax/" if "dmax" in os.listdir() else "") + "data/gdrive_ids.json") as f:
    ids = json.load(f)

DATASET_CACHE = os.path.join(CACHE, "datasets")
os.makedirs(DATASET_CACHE, exist_ok=True)

__all__ = ["DATASET_CACHE", "init", "available_datasets", "url", "download"]


def init():
    """initialize the resource for interacting with Google Drive API."""
    service = build_service()
    return service


def available_datasets(degradation: Optional[str] = None, verb: bool = True) -> List[str]:
    """available datasets to download on the given degradation"""
    datasets = []

    for task, all_ds in ids.items():
        if degradation is None or degradation == task:
            datasets.extend(all_ds.keys())
            if verb: print(f"{task}:")
            for ds_name, ds in all_ds.items():
                if verb: print(f"{ds_name} ({', '.join(ds.keys())})")
            if verb: print("")
    return datasets


def url(dataset_name: str, verb: bool = True) -> Dict[Literal["train", "validation", "test"], str]:
    """dataset url in google drive"""
    if dataset_name not in available_datasets(verb=False):
        raise ValueError("dataset not supported.")

    for all_ds in ids.values():
        if dataset_name in all_ds.keys():
            ds = all_ds[dataset_name]
            if verb:
                print(f"{dataset_name}:")
                for split, id in ds.items():
                    print(f"{split}:", f"https://drive.google.com/file/d/{id}/view?usp=sharing")
            return ds


def download(
        dataset_name: str,
        verb: bool = True,
        force_redownload: bool = False,
        absolute_url: Optional[str] = None
):
    """download given dataset to ~/.cache"""
    path = os.path.join(DATASET_CACHE, dataset_name)
    if os.path.exists(path) and not force_redownload:
        if verb: print(f"found cached dataset at {path}")
        return

    if dataset_name not in available_datasets(verb=False) and absolute_url is None:
        raise ValueError(f"""
        dataset not supported for automatic download.
        Please download dataset manually under {path}
        """)

    is_drive_id = absolute_url is not None and "https" not in absolute_url
    if absolute_url and not is_drive_id:
        service = None
        split, extension = os.path.basename(absolute_url).split(".", 1)
        url_map = {split: absolute_url}
        extension = ("." + extension)
    else:
        url_map = url(dataset_name, verb=verb) if not absolute_url else {dataset_name: absolute_url}
        service = init()
        extension = ".zip"

    if verb: print(f"will download {dataset_name} to {path}:")
    for split, abs_url in url_map.items():
        split_path = os.path.join(path+"_download", split)
        os.makedirs(split_path, exist_ok=True)
        download_file(service, split_path+extension, abs_url, verbose=verb)
        shutil.unpack_archive(split_path+extension, split_path)
    os.rename(path+"_download", path)


if __name__ == '__main__':
    jsonargparse.CLI()
