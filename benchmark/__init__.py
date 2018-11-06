import logging

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .dataset import Dataset
from .dataset import LibsvmDataset
from .model import Model
from .utils import AttributeDict

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

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

_selection_methods = AttributeDict()
_selection_methods.grid_search_cv = GridSearchCV


def get_dataset(dataset: str) -> Dataset:
    """Get dataset by its name."""
    dataset = dataset.lower()

    if dataset not in _datasets:
        raise ValueError(f'Dataset {dataset} does not exist!')

    return _datasets[dataset]


def get_model(model: str) -> Model.__class__:
    """Get model by its name."""
    model = model.lower()

    if model not in _models:
        raise ValueError(f'Model {model} does not exist!')

    return _models[model]


def get_selection_method(model_selection: str):
    """Get model selection method by its name."""
    model_selection = model_selection.lower()

    if model_selection not in _selection_methods:
        raise ValueError(f'Selection method {model_selection} does not exist!')

    return _selection_methods[model_selection]


__all__ = ['Dataset', 'Model', 'get_dataset', 'get_model', 'get_selection_method']
