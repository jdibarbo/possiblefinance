# Environment
To create the environment required to run the notebooks, in the terminal/cmd: 
- go to the folder that contains the repository
- run conda env create -f environment.yml
- run python -m ipykernel install --user --name=possible to install in jupyter

# Objective 

The main questions to answer from the data are:
1. How to identify applicants that are not going to perform well based on the features computed from transaction data
2. To target a particular risk/default rate how do we set that given the mix of applications

# Overview 
## **ITEM 1**
To answer item 1 I trained a classification model, where the target was the 'status' feature from the *approved_application_data.csv*. Applications where this feature was 0 have performed badly. 
The *applications_rejected* file can't be used in modelling as I don't have the ground truth value for these applicants.

Steps taken to train model:<br> 
- Exploratory Analysis on the data.
- **Feature selection**<br> 
    - Trained an inital estimator to use Recursive Feature elimination and select only the most important features, and compare to a Logistic Regression model with L1 penalty.   
    - Trained a Decision Tree model to use as a descriptive model, and see what are some of the rules that split the data, as well as confirm the results obtained in the previous step.<br>

- **Model training**<br>

Using the features obtained from the previous step, I trained a classification model and tried to optimize it using grid search, and with it identify the most significant features in it: 
 
- I trained two classification models (*XGBoost, Random Forest*), choose the one that fits better and
- From the selected model, see the SHAP values for the features to understand how they affect the prediction, in this case focusing on the poorly performing applicants. 

**Results**<br>
The model selected was the Random Forest model, as the precision metric was higher. Due to the time constraints, this model is preliminary. The results are not good but could be improved. Possible improvements are listed below in model results.


## **ITEM 2**
    
For **item 2**, I use the model trained for item 1, as what was asked is esentially the prediction of the probability of default, where I assume applicants in good standing (status = 1) have not defaulted, while applicants in bad standing (status = 0) have. The default rate will be the probabilty of each applicant to default as determined by the chosen model, and the threshold will be selected depending on the tolerance to risk.     

I then use the model in the rejected applications file, to see how that would have looked like under different probability thresholds. 
