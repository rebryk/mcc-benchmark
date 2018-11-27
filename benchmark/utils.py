import json
from datetime import datetime
from pathlib import Path

import requests


def download_file(url: str, path: Path):
    """Download file from the given url and store it to path."""
    response = requests.get(url, stream=True)
    with path.open('wb') as file:
        for data in response.iter_content():
            file.write(data)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def parse_params(params: str) -> dict:
    """Parse the given parameters to dictionary."""
    return json.loads(params.replace('\'', '\"')) if params else dict()


def eval_params(params: dict) -> dict:
    """Evaluate values if they are not string."""
    return {key: eval(value) if isinstance(value, str) else value for key, value in params.items()}


class Timer:
    def __init__(self, description: str = None, logger=None):
        self.description = description
        self.logger = logger
        self.time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.time = datetime.now() - self.start_time

        if self.description is not None and self.logger is not None:
            self.logger.info(f'{self.description}: {self.time} ({self.total_seconds():.0f} seconds)')

    def total_seconds(self) -> int:
        return int(self.time.total_seconds())
