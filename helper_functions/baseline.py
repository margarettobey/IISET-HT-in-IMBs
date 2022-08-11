# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 23:38:07 2022

@author: mgtobey
"""

import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score

def baselines(full_data, skf):
    def get_results(clf, X, y, method,cv_number,results):
        probs = clf.predict_proba(X)[:,1]
        predictions = clf.predict(X)
        test_conf = confusion_matrix(y,predictions)
        fpr, tpr, _ = roc_curve(y,probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        results = results.append(pd.DataFrame(columns = cols,\
                data = [[method, cv_number,test_conf[0,0],test_conf[0,1],\
                test_conf[1,0],test_conf[1,1], precision_score(y, predictions),\
                recall_score(y, predictions),f1_score(y,predictions),roc_auc_score(y,probs),fpr,tpr,interp_tpr]]))
        return results
    
    y = full_data[['label']]
    X = full_data.drop(columns = ['label'])
    X_norm = (X-X.min(0))/(X.max(0)-X.min(0))
    X_norm = add_constant(X_norm)
    
    cols = ['method', 'cv_number','TN','FP','FN','TP','precision','recall','f1_score','auc','fpr','tpr','interp_tpr']
    results = pd.DataFrame(columns = cols)
    mean_fpr = np.linspace(0, 1, 100)

    # gives 5 splits of train and test
    cv_number = 0
    for train_index, test_index in skf.split(X_norm,y):
        cv_number +=1
        X_train, X_test = X_norm.loc[train_index,:], X_norm.loc[test_index,:]
        y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]
        
        # logistic regression
        method = 'log_reg'
        classifier = LogisticRegression(class_weight = {0:1,1:4})
        clf = classifier.fit(X_train,y_train.values.ravel())
        results = get_results(clf,X_test,y_test,method,cv_number,results)
        
        # random forest
        method = 'rand_forest'
        classifier = RandomForestClassifier(random_state = 0)
        clf = classifier.fit(X_train,y_train.values.ravel())
        results = get_results(clf,X_test,y_test,method,cv_number,results)
    
        # SVM
        method = 'SVM'
        classifier = SVC(kernel = 'rbf', class_weight = {0:1,1:4}, probability = True)
        clf = classifier.fit(X_train,y_train.values.ravel())
        results = get_results(clf,X_test,y_test,method,cv_number,results)
    
        # bayes
        method = 'naive_bayes'
        classifier = naive_bayes.BernoulliNB()
        clf = classifier.fit(X_train,y_train.values.ravel())
        results = get_results(clf,X_test,y_test,method,cv_number,results)
    
    # aggregate
    results = results.reset_index(drop=True)
    tprs = results[['method','interp_tpr']]
    tprs_mean = tprs.groupby('method')['interp_tpr'].apply(np.mean).reset_index()
    results[['TP','FN','FP','TN']] = results[['TP','FN','FP','TN']].astype('float')
    results_agg = results.groupby('method').agg('mean').merge(tprs_mean,on='method')
    results_agg['AUC Range'] = results[['method','auc']].groupby('method').agg(np.ptp).values
    results_agg = results_agg.set_index('method')
    
    return results, results_agg

