import logging

from .dataset import datasets
from .model import Model

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

__all__ = ['Model', 'datasets']
