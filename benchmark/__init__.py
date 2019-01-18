import logging

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier
from xgboost import XGBClassifier

from .dataset import Dataset
from .dataset import LibsvmDataset
from .dataset import ImageSegmentation
from .dataset import Covertype
from .dataset import WinequalityWhite
from .dataset import Abalone
from .model import FMCBoosting
from .model import Model
from .utils import AttributeDict

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

_datasets = AttributeDict()
_datasets.iris = LibsvmDataset('iris.scale')
_datasets.wine = LibsvmDataset('wine.scale')
_datasets.glass = LibsvmDataset('glass.scale')
_datasets.aloi100 = LibsvmDataset('aloi.scale.bz2', n_classes=100)
_datasets.aloi250 = LibsvmDataset('aloi.scale.bz2', n_classes=250)
_datasets.aloi500 = LibsvmDataset('aloi.scale.bz2', n_classes=500)
_datasets.aloi1000 = LibsvmDataset('aloi.scale.bz2', n_classes=1000)
_datasets.aloi100i = LibsvmDataset('aloi.scale.bz2', n_classes=100, imbalanced=True)
_datasets.aloi250i = LibsvmDataset('aloi.scale.bz2', n_classes=250, imbalanced=True)
_datasets.aloi500i = LibsvmDataset('aloi.scale.bz2', n_classes=500, imbalanced=True)
_datasets.aloi1000i = LibsvmDataset('aloi.scale.bz2', n_classes=1000, imbalanced=True)
_datasets.segment = LibsvmDataset('segment.scale')
_datasets.letter = LibsvmDataset('letter.scale', 'letter.scale.t')
_datasets.news20 = LibsvmDataset('news20.scale.bz2', 'news20.t.scale.bz2')
_datasets.mnist = LibsvmDataset('mnist.scale.bz2', 'mnist.scale.t.bz2')
_datasets.pendigits = LibsvmDataset('pendigits', 'pendigits.t')
_datasets.image_segmentation = ImageSegmentation()
_datasets.covertype = Covertype()
_datasets.winequality_white = WinequalityWhite()

# Some classes consist of only one sample
# _datasets.abalone = Abalone()

_models = AttributeDict()
_models.svm = SVC
_models.gnb = GaussianNB
_models.knc = KNeighborsClassifier
_models.rfc = RandomForestClassifier
_models.dtc = DecisionTreeClassifier
_models.xgb = XGBClassifier
_models.cat = CatBoostClassifier
_models.lgbm = LGBMClassifier
_models.elm = ELMClassifier
_models.fmcb = FMCBoosting

_selection_methods = AttributeDict()
_selection_methods.grid_search_cv = GridSearchCV
_selection_methods.randomized_search_cv = RandomizedSearchCV


def get_dataset(dataset: str) -> Dataset:
    """Get dataset by its name."""
    dataset = dataset.lower()

    if dataset not in _datasets:
        raise ValueError(f'Dataset {dataset} does not exist!')

    return _datasets[dataset]


def get_model_class(model: str) -> Model.__class__:
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


__all__ = ['Dataset', 'Model', 'get_dataset', 'get_model_class', 'get_selection_method']
