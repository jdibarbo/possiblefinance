# Environment
To create the environment required to run the notebooks, in the terminal/cmd: 
- go to the folder that contains the repository
- run conda env create -f environment.yml
- run python -m ipykernel install --user --name=possible to install in jupyter

# Objective 

The main questions to answer from the data are:
1. How to identify applicants that are not going to perform well based on the features computed from transaction data
2. To target a particular risk/default rate how do we set that given the mix of applications

For **item 1**, I take into account only the *applications_approved* file, and build classification model that allows me to see what the most important features are in determining poorly perfromant applicants. Here, the target variable is be **status**. 

The *applications_rejected* file can't be used in modelling as I don't have the ground truth value for these applicants.

Steps:<br> 
- Exploratory Analysis on the data.
- train a logistic regression with L1 penalty, and a Random Forest model and use one of them as feature selector. 
- train two classification models: an XGBoost and a Random Forest model, both for classification, to see what are the important features to identify applicants who will perform better. 
- get SHAP values for the selected model to see the impact on the predictions and see which features help identify the poorly performing applicants.

Conclusion: 
The final selected model was the Random Forest model, as the precision metric was higher.

For **item 2**, I use the model trained for item 1, as what was asked is esentially the prediction of the probability of default, where I assume applicants in good standing (status = 1) have not defaulted, while applicants in bad standing (status = 0) have. The default rate will be the probabilty of each applicant to default as determined by the chosen model, and the threshold will be selected depending on the tolerance to risk.     
