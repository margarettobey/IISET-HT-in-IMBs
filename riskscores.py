# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:10:06 2022
 
@author: mgtobey
"""
#%% packages
import pandas as pd
import os
import pickle

from sklearn.model_selection import StratifiedKFold

import riskslim
from helper_functions.riskscore_train import riskscore_train#, getScores

#%% input parameters
# to reproduce results 3600, 10, 0.5
runtime = 6
size = 10 # (max features selected by risk score model)
thresh = 0.5  # (classification threshold)

#%% final set of features and undersampled version
full_data = pd.read_csv('selected_features_1612and403.csv',index_col=0)

#%% get data and labels
y = full_data[['label']]
X = full_data.drop(columns = ['label'])

#%% folders, create results folder, if already created, comment next line or delete folder
os.mkdir('riskscore_results')
path = os.getcwd()+r'\riskscore_results'
# navigate to folder where risk-slim package is stored, update if needed
os.chdir(r"C:\Users\mgtobey\risk-slim")

#%% 5 fold CV
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle = True)
# loops through each of the 5 splits of train and test
cv_number = 0
for train_index, test_index in skf.split(X,y):
    cv_number +=1
    
    # get fold train and test data
    X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
    y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]
    data_train = y_train.merge(X_train,left_index=True, right_index=True)
    data_test = y_test.merge(X_test,left_index=True, right_index=True)
    
    # temp output to load via risk slim package
    data_train.to_csv(r'cv'+str(cv_number)+'_riskscore_data_train.csv',index=False)
    data_test.to_csv('cv'+str(cv_number)+'_riskscore_data_test.csv',index=False)

    # train risk scores on train data, load data and then train
    train_data_dict = riskslim.load_data_from_csv(dataset_csv_file = r'cv'+str(cv_number)+'_riskscore_data_train.csv')
    model_info,mip_info, lcpa_info = riskscore_train(train_data_dict,size,runtime)

    # display resulting model
    table = riskslim.print_model(model_info['solution'], train_data_dict)
    # can optionally choose to export text file with resulting model for each CV fold
    # table_txt = table.get_string()
    # result_filename = path+r'\cv' + str(cv_number) + '_results.txt'
    # with open(result_filename, "w") as text_file:
    #     text_file.write('runtime: ' + str(runtime) + '\n')
    #     text_file.write('max_size: ' + str(size)+ '\n')
    #     text_file.write(table_txt)
    #     text_file.write('\n')
    #     text_file.write(str(model_info))
    
    # save model and output test results
    model_filename = path+r'\cv' + str(cv_number) + '_model.sav'
    pickle.dump(model_info['solution'],open(model_filename,'wb'))

    # delete csvs
    os.remove(r'cv'+str(cv_number)+'_riskscore_data_train.csv')
    os.remove(r'cv'+str(cv_number)+'_riskscore_data_test.csv')

#%% run on full data 
full_data.to_csv(r'fulldata_riskscore_data_train.csv',index=False)

# train risk scores on full data
full_data_dict = riskslim.load_data_from_csv(dataset_csv_file = r'fulldata_riskscore_data_train.csv') 
model_info,mip_info, lcpa_info  = riskscore_train(full_data_dict,size,runtime)

# print results to text file
table = riskslim.print_model(model_info['solution'], full_data_dict)
table_txt = table.get_string()

result_filename = path+r'\full_results.txt'
with open(result_filename, "w") as text_file:
    text_file.write('runtime: ' + str(runtime) + '\n')
    text_file.write('max_size: ' + str(size)+ '\n')
    text_file.write(table_txt)
    text_file.write('\n')
    text_file.write(str(model_info))

# save model and output training results
model_filename = path+r'\full_data_model.sav'
pickle.dump(model_info['solution'],open(model_filename,'wb'))

# delete csvs
os.remove(r'fulldata_riskscore_data_train.csv')
