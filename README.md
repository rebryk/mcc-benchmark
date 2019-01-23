# Benchmarking of MultiClass Classification models
[![Build Status](https://travis-ci.com/rebryk/mcc-benchmark.svg?branch=master)](https://travis-ci.com/rebryk/mcc-benchmark)

The purpose of this tool is benchmarking of MultiClass Classification models.

## Installation
Python 3 is required.
```
git clone https://github.com/rebryk/mcc-benchmark.git
cd mcc-benchmark
pip3 install -r requirements.txt
```

## Usage
You can use `eval.py` to train a model on a dataset, but you should specify some options. <br>
Currently we support the following options:
* `--model MODEL` - the name of the model to train
* `--params PARAMS` - parameters of the model in json format
* `--dataset DATASET` - dataset name
* `--valid_size VALID_SIZE` - represents the proportion of the dataset to include in the valid split
* `--test_size TEST_SIZE` - represents the proportion of the dataset to include in the test split
* `--n_runs N_RUNS` - the number of train/valid/test splits 
* `--selection SELECTION` - method of hyperparameter tuning
* `--selection_params SELECTION_PARAMS` - parameters of the selection method in json format
* `--param_grid PARAM_GRID` - enables searching over any sequence of parameter settings

### Example 1
To train `svm` model with default parameters on `letter` dataset, you can run the folliwing command: <br>
`python eval.py --model svm --dataset letter` 

```
11/06/2018 01:31:58 PM - Experiment - INFO - Model: svm
11/06/2018 01:31:58 PM - Experiment - INFO - Model parameters: default
11/06/2018 01:31:58 PM - Experiment - INFO - Loading letter dataset...
11/06/2018 01:31:58 PM - Experiment - INFO - Train size: 15000
11/06/2018 01:31:58 PM - Experiment - INFO - Test size: 5000
11/06/2018 01:31:58 PM - Experiment - INFO - Run #1
11/06/2018 01:31:58 PM - Experiment - INFO - Training the model...
11/06/2018 01:32:02 PM - Experiment - INFO - Training time: 0:00:03.916086 (3 seconds)
11/06/2018 01:32:12 PM - Experiment - INFO - Train score:       0.8239
11/06/2018 01:32:15 PM - Experiment - INFO - Prediction time (test): 0:00:03.359526 (3 seconds)
11/06/2018 01:32:15 PM - Experiment - INFO - Test score:        0.8156
11/06/2018 01:32:15 PM - Experiment - INFO - Total time: 0:00:16.757443 (16 seconds)
```

### Example 2
You can specify some parameters of the model manually: <br>
`python eval.py --model svm --params "{'gamma': 'auto', 'C': 10}" --dataset letter`

```
11/06/2018 01:33:46 PM - Experiment - INFO - Model: svm
11/06/2018 01:33:46 PM - Experiment - INFO - Model parameters: {'gamma': 'auto', 'C': 10}
11/06/2018 01:33:46 PM - Experiment - INFO - Loading letter dataset...
11/06/2018 01:33:46 PM - Experiment - INFO - Train size: 15000
11/06/2018 01:33:46 PM - Experiment - INFO - Test size: 5000
11/06/2018 01:33:46 PM - Experiment - INFO - Run #1
11/06/2018 01:33:46 PM - Experiment - INFO - Training the model...
11/06/2018 01:33:48 PM - Experiment - INFO - Training time: 0:00:02.317413 (2 seconds)
11/06/2018 01:33:54 PM - Experiment - INFO - Train score:       0.9027
11/06/2018 01:33:57 PM - Experiment - INFO - Prediction time (test): 0:00:02.180381 (2 seconds)
11/06/2018 01:33:57 PM - Experiment - INFO - Test score:        0.8966
11/06/2018 01:33:57 PM - Experiment - INFO - Total time: 0:00:11.010437 (11 seconds)
```

If the dataset does not have separate test part, you can use `test_size` option to specify the proportion of the dataset to include in the test.

### Example 3
It's possible to use [grid search cross-validation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to tune hyperparameters:
```
python eval.py --model svm --params "{'gamma': 'auto'}" --dataset letter --selection grid_search_cv --selection_params "{'cv': 3}" --param_grid "{'C': [5, 10, 20]}"
```

```
11/06/2018 01:34:33 PM - Experiment - INFO - Model: svm
11/06/2018 01:34:33 PM - Experiment - INFO - Model parameters: {'gamma': 'auto'}
11/06/2018 01:34:33 PM - Experiment - INFO - Selection method: grid_search_cv
11/06/2018 01:34:33 PM - Experiment - INFO - Selection parameters: {'cv': 3}
11/06/2018 01:34:33 PM - Experiment - INFO - Parameters grid: {'C': [5, 10, 20]}
11/06/2018 01:34:33 PM - Experiment - INFO - Loading letter dataset...
11/06/2018 01:34:33 PM - Experiment - INFO - Train size: 15000
11/06/2018 01:34:33 PM - Experiment - INFO - Test size: 5000
11/06/2018 01:34:33 PM - Experiment - INFO - Run #1
11/06/2018 01:34:33 PM - Experiment - INFO - Searching the best parameters...
11/06/2018 01:35:25 PM - Experiment - INFO - Searching time: 0:00:51.572077 (51 seconds)
11/06/2018 01:35:25 PM - Experiment - INFO - Grid scores on validate set:
11/06/2018 01:35:25 PM - Experiment - INFO - Valid score: 0.8613 (+/-0.002) for {'C': 5}
11/06/2018 01:35:25 PM - Experiment - INFO - Valid score: 0.8797 (+/-0.003) for {'C': 10}
11/06/2018 01:35:25 PM - Experiment - INFO - Valid score: 0.8952 (+/-0.002) for {'C': 20}
11/06/2018 01:35:25 PM - Experiment - INFO - Best parameters: {'C': 20}
11/06/2018 01:35:25 PM - Experiment - INFO - Training the model...
11/06/2018 01:35:27 PM - Experiment - INFO - Training time: 0:00:02.140764 (2 seconds)
11/06/2018 01:35:32 PM - Experiment - INFO - Train score:       0.9231
11/06/2018 01:35:34 PM - Experiment - INFO - Prediction time (test): 0:00:01.808641 (1 seconds)
11/06/2018 01:35:34 PM - Experiment - INFO - Test score:        0.9090
11/06/2018 01:35:34 PM - Experiment - INFO - Total time: 0:01:01.222041 (61 seconds)
```

Also you can use `range`, `np.linspace` in `param_grid`: <br>
`--param_grid "{'C': 'range(1, 10)'}"` <br>
`--param_grid "{'C': '2 ** np.linspace(8, 11, 15)'}"`

### Example 4
If you want to run the Factorized MultiClass Boosting method, you can run the following command: <br>
```
python eval.py --model fmcb --dataset letter --params "{'n_iter': 5000, 'depth': 8, 'lr': 1}"
```

```
11/06/2018 01:47:04 PM - Experiment - INFO - Model: fmcb
11/06/2018 01:47:04 PM - Experiment - INFO - Model parameters: {'n_iter': 5000, 'depth': 8, 'lr': 1}
11/06/2018 01:47:04 PM - Experiment - INFO - Loading letter dataset...
11/06/2018 01:47:04 PM - Experiment - INFO - Train size: 15000
11/06/2018 01:47:04 PM - Experiment - INFO - Test size: 5000
11/06/2018 01:47:04 PM - Experiment - INFO - Run #1
11/06/2018 01:47:04 PM - Experiment - INFO - Training the model...
11/06/2018 01:48:42 PM - Experiment - INFO - Training time: 0:01:37.484042 (97 seconds)
11/06/2018 01:48:52 PM - Experiment - INFO - Train score:       0.9952
11/06/2018 01:48:57 PM - Experiment - INFO - Prediction time (test): 0:00:04.931633 (4 seconds)
11/06/2018 01:48:57 PM - Experiment - INFO - Test score:        0.9562
11/06/2018 01:48:57 PM - Experiment - INFO - Total time: 0:01:53.261572 (113 seconds)
```

## Models
Currently we support the following models:
* `svm` - [C-Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* `gnb` - [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* `knc` - [K-Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) 
* `rfc` - [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* `dtc` - [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* `xgb` - [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier/)
* `cat` - [CatBoost Classifier](https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier-docpage/)
* `lgbm` - [LGBM Classifier](https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMClassifier)
* `elm` - [ELM Classifier](https://github.com/dclambert/Python-ELM)
* `fmcb` - Factorized MultiClass Boosting

## Datasets
Currently we support some multiclass classification datasets from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html):
* `iris`
* `wine`
* `glass`
* `aloi100` - aloi dataset with 100 random classes.
* `aloi250` - aloi dataset with 250 random classes.
* `aloi500` - aloi dataset with 500 random classes.
* `aloi1000` - original aloi dataset.
* `segment`
* `letter`
* `new20`
* `mnist`
* `pendigits`

**UPD.** Added support of some multiclass classification datasets from [UCI](http://archive.ics.uci.edu/ml/index.php):
* [`image_segmentation`](http://archive.ics.uci.edu/ml/datasets/image+segmentation)
* [`covertype`](https://archive.ics.uci.edu/ml/datasets/Covertype)
* [`winequality_white`](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* [`abalone`](https://archive.ics.uci.edu/ml/datasets/abalone)

## License
[MIT](LICENSE)
