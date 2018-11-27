import logging

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier
from xgboost import XGBClassifier

from .dataset import Dataset
from .dataset import LibsvmDataset
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
_datasets.segment = LibsvmDataset('segment.scale')
_datasets.letter = LibsvmDataset('letter.scale', 'letter.scale.t')
_datasets.news20 = LibsvmDataset('news20.scale.bz2', 'news20.t.scale.bz2')

_models = AttributeDict()
_models.svm = SVC
_models.gnb = GaussianNB
_models.knc = KNeighborsClassifier
_models.rfc = RandomForestClassifier
_models.dtc = DecisionTreeClassifier
_models.one_vs_rest_lgbm = lambda *args, **kwargs: OneVsRestClassifier(LGBMClassifier(*args, **kwargs))
_models.one_vs_rest_gbc = lambda *args, **kwargs: OneVsRestClassifier(GradientBoostingClassifier(*args, **kwargs))
_models.one_vs_rest_cat = lambda *args, **kwargs: OneVsRestClassifier(CatBoostClassifier(*args, **kwargs))
_models.one_vs_rest_xgb = lambda *args, **kwargs: OneVsRestClassifier(XGBClassifier(*args, **kwargs))
_models.elm = ELMClassifier
_models.fmcb = FMCBoosting

_selection_methods = AttributeDict()
_selection_methods.grid_search_cv = GridSearchCV


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
