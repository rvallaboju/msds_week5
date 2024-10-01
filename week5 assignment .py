#!/usr/bin/env python
# coding: utf-8

# # DS Automation Assignment
# Using our prepared churn data from week 2:
# 
# use pycaret to find an ML algorithm that performs best on the data
# Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
# save the model to disk
# create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
# your Python file/function should print out the predictions for new data (new_churn_data.csv)
# the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# test your Python module and function with the new data, new_churn_data.csv
# write a short summary of the process and results at the end of this notebook
# upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox
# Optional challenges:
# 
# return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
# use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
# create a class in your Python module to hold the functions that you created
# accept user input to specify a file using a tool such as Python's input() function, the click package for command-line arguments, or a GUI
# Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.

# In[10]:


import pandas as pd
data=pd.read_csv('C:/Users/DELL/Downloads/ruu_churn2.csv',index_col='customerID')
data


# In[3]:


from pycaret.classification import setup, compare_models, predict_model, save_model, load_model


# In[6]:


automl = setup(data, target='Churn')


# In[18]:


correct_model = compare_models(sort='AUC')


# In[19]:


correct_model = compare_models(sort='Recall')


# In[20]:


correct_model


# In[23]:


data.iloc[-3:-1].shape


# In[26]:


predict_model(correct_model, data.iloc[-3:-1])


# In[22]:


save_model(correct_model, 'correct_model')


# In[17]:


from pycaret.classification import load_model, predict_model

def load_data(filepath):
   
    df = pd.read_csv(filepath, index_col='customerID')
    return df

def make_predictions(df):
   
    model = load_model('best_model') 
    predictions = predict_model(model, data=df)

    print("Predictions DataFrame Columns:")
    print(predictions.columns)

    if 'prediction_label' in predictions.columns:
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True)
        
        predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
        
        return predictions[['Churn_prediction', 'prediction_score']]
    else:
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")

if __name__ == "__main__":
    new_data = load_data('C:/Users/DELL/Downloads/new_churn_data.csv')
    predictions = make_predictions(new_data)
    true_values = [1, 0, 0, 1, 0]
    
    print('Predictions:')
    print(predictions)
    print('True Values:')
    print(true_values)
    print('Predictions:')
    print(predictions)


    

Analysis:

PyCaret's classification module automates essential machine learning tasks, simplifying workflows. The setup function handles preprocessing like imputation, encoding, scaling, and data splitting. compare_models ranks algorithms, while predict_model generates predictions. save_model and load_model ensure model persistence. This automation accelerates customer churn prediction, enabling faster experimentation and easier model management.

The code compare_models(sort='AUC') identifies the model with the best AUC score, balancing both false positives and false negatives. On the other hand, compare_models(sort='Recall') focuses on selecting the model that reduces false negatives, ensuring that all positive cases are correctly detected. This approach is especially important in situations like medical diagnosis or fraud detection, where missing a positive case can have significant consequences.

data.iloc[-3:-1].shape retrieves the dimensions of the second-to-last and third-to-last rows in the dataset.
predict_model(correct_model, data.iloc[-3:-1]) makes predictions on those rows using the trained model.
save_model(correct_model, 'correct_model') stores the model for future use, ensuring it can be reused or deployed.

In conclusion,this suggests that the model did not detect any instances of churn, underscoring the necessity for further improvements to enhance its predictive accuracy. Ultimately, this process demonstrates the stages of training a model, saving it, and utilizing it for churn predictions.


















