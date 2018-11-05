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
