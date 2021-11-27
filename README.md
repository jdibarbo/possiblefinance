# Environment
To create the environment required to run the notebooks, in the terminal/cmd: 
- go to the folder that contains the repository
- run conda env create -f environment.yml

# Objective 

The main questions to answer from the data are:
1. How to identify applicants that are not going to perform well based on the features computed from transaction data
2. To target a particular risk/default rate how do we set that given the mix of applications

For **item 1**, I'll take into account only the *applications_approved* file, and build classification model that will allow me to see what the most important features are in determining poorly perfromant applicants. Here, the target variable will be **status**. 

The *applications_rejected* file can't be used in modelling as I don't have the ground truth value for these applicants.

I will: 

- do an Exploratory Analysis on the data
- train a logistic regression with L1 penalty, and a Random Forest model and use one of them as feature selector. 
- train two classification models: an XGBoost and a Random Forest model, both for classification, to see what are the important features to identify applicants who will perform better. 

For **item 2**, I will use the model trained for item 1, as what is asked is esentially the prediction of the probability of default, where I assume applicants in good standing (status = 1) have not defaulted, while applicants in bad standing (status = 0) have. Therefore, the default rate will be the probabilty of each applicant to default as determined by the chosen model.    

