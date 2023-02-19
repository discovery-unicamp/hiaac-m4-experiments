import logging
import os
import platform
import re
import socket
import time
from typing import List
import uuid
from pathlib import Path

import psutil
import yaml
from librep.config.type_definitions import PathLike
from librep.datasets.multimodal.multimodal import MultiModalDataset


class catchtime:
    """Utilitary class to measure time in a `with` python statement."""

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.e = time.time()

    def __float__(self):
        return float(self.e - self.t)

    def __coerce__(self, other):
        return (float(self), other)

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return str(float(self))


def load_yaml(path: PathLike) -> dict:
    """Utilitary function to load a YAML file.

    Parameters
    ----------
    path : PathLike
        The path to the YAML file.

    Returns
    -------
    dict
        A dictionary with the YAML file content.
    """
    path = Path(path)
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def get_sys_info():
    try:
        info = {}
        info["platform"] = platform.system()
        info["platform-release"] = platform.release()
        info["platform-version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip-address"] = socket.gethostbyname(socket.gethostname())
        info["mac-address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
        info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["total_cores"] = psutil.cpu_count(logical=True)
        return info
    except Exception as e:
        logging.exception("Error getting info")
        return dict()


def multimodal_multi_merge(datasets: List[MultiModalDataset]) -> MultiModalDataset:
    merged = datasets[0]
    for dataset in datasets[1:]:
        merged = merged.merge(dataset)
    return merged