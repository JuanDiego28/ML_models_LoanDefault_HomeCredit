# Sprint project 02
> Home Credit Default Risk

## The Business problem

This is a binary Classification task: we want to predict whether the person applying for a home credit will be able to repay their debt or not. Our model will have to predict a 1 indicating the client will have payment difficulties: he/she will have late payment of more than X days on at least one of the first Y installments of the loan in our sample, 0 in all other cases.

We will use [Area Under the ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es_419) as the evaluation metric, so our models will have to return the probabilities that a loan is not paid for each input data.

## About the data

The original dataset is composed of multiple files with different information about loans taken. this dataset participated in a kaggle competition under the name: "Home Credit Default Risk",  In this project, we will work exclusively with the primary files: `application_train_aai.csv` and `application_test_aai.csv`.


## Home Credit
"Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities."(https://www.kaggle.com/competitions/home-credit-default-risk/overview)

Data will be automatically downloaded from the main note book

## Technical aspects

The technologies involved are:
- Python as the main programming language
- Pandas for consuming data from CSVs files
- Scikit-learn for building features and training ML models
- Matplotlib and Seaborn for the visualizations
- Jupyter notebooks to make the experimentation in an interactive way


