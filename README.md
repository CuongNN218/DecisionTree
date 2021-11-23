# DecisionTree
Implementation of Decision Tree Algorithm using Gini Index

## Setup
To install python packages for running code, run the following command:
- pip install -r requirements.txt

## Preprocessing
Running the following command to preprocess dataset:\
- For train dataset run ``python3 preprocessing.py --input_file adult.train --output_file train.csv``
- For test dataset run ``python3 preprocessing.py --input_file adult.test --output_file test.csv``

## Training
To run decision tree algorithm, run the following command:\
``python3 dtbuild.py --train_file <path/to/training_data> --model_file <path/to/save_model> --min_freq <min_freq>``
## Prediction
To get result for test set, using this command:\
``python3 dtclassify.py --model_file <path/to/model_file> --test_file <path/to/test/file> --predictions <name of prediction file>``
## Evaluate
To evaluate your result on test set, using this command:\
``python3 dtevaluate --pred <path/to/prediction_file>``