# Prediction of Water Potability using AutoML and HyperDrive

This project demonstrates how to use AutoML and HyperDrive to train models that predict water potability using an external dataset not directly availabled in AzureML.

The best model trained with AutoML and the best model trained with HyperDrive are compared using the highest AUC weighted metric, and the model with the best performance is deployed as a web service in an Azure Container Instance.

Once the best model is deployed, it is tested by making a web service call to predict water potability using a sample dataset.


## Project Set Up and Installation
To set up the project, we need to create a Compute Instance and open the Jupyter interface. In Jupyter, we need to upload the following files:
- automl.ipynb
- hyperparameter_tuning.ipynb
- train_randomforest.py

We need to also create a "data" folder, and upload the water_potability.csv file provided as part of this project.

## Dataset

### Overview
I am using the Water Potability dataset obtained from Kaggle: https://www.kaggle.com/nickyudin/waterpotability

This dataset was manually downloaded from Kaggle as a CSV file and it was stored in the data folder of the submission Zip file, and should also be uploaded in a "data" folder alongside (data folder at same level of) the Jupyter notebooks.


### Task
The water potability dataset contains 9 decimal columns that will be used as features for the model, and 1 integer column ("Potability") that will be the label or dependent variable to predict.

This will be a classification task as the model will predict the Potability variable which takes the value 0 (non-potable) or 1 (potable)

Features:
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity

### Access
The water_potability.csv file needs to be uploaded to the "data" folder in the Compute Instance. A cell in any of the two Jupyter notebooks (automl.ipynb or hyperparameter_tuning.ipynb) will upload this data to the blob storage of the default dataset for the workspace and the dataset will be registered as a Tabular dataset sourced from the CSV file.

## Automated ML
An experiment was submitted by passing an AutoML configuration object that specified the task the model should perform ("classification"), the variable to predict ("Potability"), and the primary metric to consider to determine the best performing model ("AUC_weighted").

Other important settings included the compute target on which the experiment would run (a 10-node computer cluster previously defined in the notebook and registered in the AzureML workspace), maximum number of jobs to run in parallel (9 as I have a 10 node cluster), whether early stopping was enabled (true), and whether featurization was required (auto based on data). I also set a maximum of 60min for the experiment.

When inspecting the tasks of the AutoML experiment, I noticed that the data is imputed using the means, and I replicate this behavior on the HyperDrive experiment.


### Results

*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
A screen recording explaining the source code and demonstrating the working model as a deployed web service can be accessed through the following link:

https://youtu.be/qQYJT813QeU


## Standout Suggestions
One of the standout suggestions to enable and use Azure Application Insights was performed for this project.

The following screenshot shows that the Application Insights was enabled on the Web Service:
![websvc_insights](screenshots/10_webservice_appinsights.png)

And after submitting a web service call to predict the potability on the sample dataset, I could see that the activity was recorded in Azure Application Insights:
![appinsights](screenshots/11_appinsights.png)
