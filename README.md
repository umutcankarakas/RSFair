# RSFair

RSFair is a directed fairness testing framework machine learning models which uses OMP and K-SVD for representative sampling process of discriminative input detection. 

## Prerequisites

* Python 2.7.15
* numpy 1.14.5
* scipy 1.1.0
* scikit-learn 1.19.0

## Background
This project used [AEQUITAS](https://github.com/sakshiudeshi/Aequitas) as base study. In this repository, Fully Directed AEQUITAS, Random Sampling and our RSFair approach can be found. 
Scikit-Learn classifiers trained on two datasets: [Adult(Census)](http://archive.ics.uci.edu/ml/datasets/Adult) and [Credit](http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

## Config
The [config](config.py) file has the following data:

* dataset : The dataset used for training
* params : The number of parameters in the data
* sensitive_param: The parameter under test.
* input_bounds: The bounds of each parameter
* classifier_name: Pickled scikit-learn classifier under test (only applicable to the sklearn files)
* threshold: Discrimination threshold.
* perturbation_unit: By what unit would the user like to perturb the input in the local search.
* retraining_inputs: Inputs to be used for the retraining. Please see [this file](Retrain_Example_File.txt).

## Demo
`python <filename>`

* Training models:  eg. `python training.py`
* Discriminative input detection: eg. `python aequitas.py`,  `python randomfair.py`,  `python rsfair.py`
* Retraining set creation: eg. `python aequitas_retrain.py`,  `python randomfair_retrain.py`,  `python rsfair_retrain.py`
* AEQUITAS (old) retraining: eg. `python retrain.py`
* New retraining: eg. `python updated_retrain.py`

## Contact
* Please contact karakasu15@itu.edu.tr for any comments/questions.
