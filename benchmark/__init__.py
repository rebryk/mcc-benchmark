import logging
from pathlib import Path
from typing import Union

from sklearn.svm import SVC

from .dataset import Dataset
from .dataset import LibsvmDataset
from .model import Model
from .utils import AttributeDict

DEFAULT_DATA_FOLDER = Path('dataset')

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

_datasets = AttributeDict()
_datasets.iris = LibsvmDataset('iris.scale')
_datasets.wine = LibsvmDataset('wine.scale')
_datasets.glass = LibsvmDataset('glass.scale')
_datasets.aloi = LibsvmDataset('aloi.scale.bz2')
_datasets.cifar10 = LibsvmDataset('cifar10.bz2', 'cifar10.t.bz2')
_datasets.letter = LibsvmDataset('letter.scale', 'letter.scale.t')
_datasets.mnist = LibsvmDataset('mnist.scale.bz2', 'mnist.scale.t.bz2')
_datasets.news20 = LibsvmDataset('news20.scale.bz2', 'news20.t.scale.bz2')

_models = AttributeDict()
_models.svm = SVC


def load_dataset(dataset: str,
                 data_folder: Union[str, Path] = DEFAULT_DATA_FOLDER,
                 test_size: Union[int, float, None] = None):
    """Load dataset by its name."""
    dataset = dataset.lower()

    if dataset not in _datasets:
        raise ValueError(f'Dataset {dataset} does not exist!')

    return _datasets[dataset].load(data_folder, test_size)


def get_model(model: str) -> Model.__class__:
    """Get model by its name."""
    model = model.lower()

    if model not in _models:
        raise ValueError(f'Model {model} does not exist!')

    return _models[model]


__all__ = ['Dataset', 'Model', 'load_dataset', 'get_model']
