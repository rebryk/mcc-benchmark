# Benchmarking of MultiClass Classification models
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
* `--model MODEL` - the name of the model to train.
* `--params PARAMS` - parameters of the model in json format.
* `--dataset DATASET` - dataset name.
* `--test_size TEST_SIZE` - represents the proportion of the dataset to include in the test split.
* `--selection SELECTION` - method of hyperparameter tuning.
* `--selection_params SELECTION_PARAMS` - parameters of the selection method in json format.
* `--param_grid PARAM_GRID` - enables searching over any sequence of parameter settings.

### Example 1
To train `svm` model with default parameters on `letter` dataset, you can run the folliwing command: <br>
`python eval.py --model svm --dataset letter` 

```
11/06/2018 07:03:05 PM - eval - INFO - Use svm model
11/06/2018 07:03:05 PM - eval - INFO - Model parameters: default
11/06/2018 07:03:05 PM - eval - INFO - Loading letter dataset...
11/06/2018 07:03:05 PM - eval - INFO - Train size:	15000
11/06/2018 07:03:05 PM - eval - INFO - Test size:	5000
11/06/2018 07:03:05 PM - eval - INFO - Training the model...
11/06/2018 07:03:09 PM - eval - INFO - Training time: 0:00:03.756155 (4 seconds)
11/06/2018 07:03:18 PM - eval - INFO - Train score:	0.823
11/06/2018 07:03:21 PM - eval - INFO - Test score:	0.821
```

### Example 2
You can specify some parameters of the model manually: <br>
`python eval.py --model svm --params "{'gamma': 'auto', 'C': 10}" --dataset letter`

```
11/06/2018 07:05:06 PM - eval - INFO - Use svm model
11/06/2018 07:05:06 PM - eval - INFO - Model parameters: {'gamma': 'auto', 'C': 10}
11/06/2018 07:05:06 PM - eval - INFO - Loading letter dataset...
11/06/2018 07:05:06 PM - eval - INFO - Train size:	15000
11/06/2018 07:05:06 PM - eval - INFO - Test size:	5000
11/06/2018 07:05:06 PM - eval - INFO - Training the model...
11/06/2018 07:05:08 PM - eval - INFO - Training time: 0:00:02.048628 (2 seconds)
11/06/2018 07:05:14 PM - eval - INFO - Train score:	0.903
11/06/2018 07:05:16 PM - eval - INFO - Test score:	0.892
```

If the dataset does not have separate test part, you can use `test_size` option to specify the proportion of the dataset to include in the test.

### Example 3
It possible to use [grid search cross-validation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to tune hyperparameters:
```
python eval.py --model svm --params "{'gamma': 'auto'}" --selection grid_search_cv --selection_params "{'cv': 3}" --param_grid "{'C': [5, 10, 20]}" --dataset letter
```

```
11/06/2018 08:04:30 PM - eval - INFO - Use svm model
11/06/2018 08:04:30 PM - eval - INFO - Model parameters: {'gamma': 'auto'}
11/06/2018 08:04:30 PM - eval - INFO - Selection method parameters: {'cv': 3}
11/06/2018 08:04:30 PM - eval - INFO - Parameters grid: {'C': [5, 10, 20]}
11/06/2018 08:04:30 PM - eval - INFO - Loading letter dataset...
11/06/2018 08:04:30 PM - eval - INFO - Train size:	15000
11/06/2018 08:04:30 PM - eval - INFO - Test size:	5000
11/06/2018 08:04:30 PM - eval - INFO - Searching the best parameters...
11/06/2018 08:05:21 PM - eval - INFO - Best parameters: {'C': 20}
11/06/2018 08:05:21 PM - eval - INFO - Grid scores on validate set:
11/06/2018 08:05:21 PM - eval - INFO - Valid score: 0.856 (+/-0.006) for {'C': 5}
11/06/2018 08:05:21 PM - eval - INFO - Valid score: 0.877 (+/-0.005) for {'C': 10}
11/06/2018 08:05:21 PM - eval - INFO - Valid score: 0.896 (+/-0.004) for {'C': 20}
11/06/2018 08:05:21 PM - eval - INFO - Searching time: 0:00:50.979066 (51 seconds)
11/06/2018 08:05:21 PM - eval - INFO - Training the model...
11/06/2018 08:05:23 PM - eval - INFO - Training time: 0:00:01.904533 (2 seconds)
11/06/2018 08:05:29 PM - eval - INFO - Train score:	0.923
11/06/2018 08:05:31 PM - eval - INFO - Test score:	0.906
```

Also you can use `range` in `param_grid`: <br>
`--param_grid "{'C': 'range(1, 10)'}"`

## Models
Currently we support the following models:
* [C-Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## Datasets
Currently we support some multiclass classification datasets from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html):
* `iris`
* `wine`
* `glass`
* `aloi`
* `cifar10`
* `letter`
* `mnist`
* `new20`

## License
[MIT](LICENSE)
