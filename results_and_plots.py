# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:45:50 2022

@author: mgtobey
"""
#%% packages
import pandas as pd
import pickle
from helper_functions.riskscore_train import getScores
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from helper_functions.baseline import baselines
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

#%% threshold for classification
thresh = 0.5

#%%  results folders
rs_folder = r'riskscore_results'
dt_folder = r'decisiontree_results'

#%% read in data sets 
full_data = pd.read_csv('selected_features_1612and403.csv',index_col=0)
presamp = pd.read_csv(r'selected_features_9070and403.csv',index_col=0)
# cross validation split
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle = True)

#%% read in decision tree results
dt_results = pd.read_csv(dt_folder+r'\[Revision]5FoldSCV_prediction_info.csv')

#%% read risk score files back in and get risk score results
results_rs = pd.DataFrame(columns = ['cv_number','TN','FP','FN','TP','precision','recall','f1_score','auc','AUC Range'])
auc_table = pd.DataFrame(columns=['cv_number', 'fpr','tpr','auc'])

y = full_data[['label']]
X = full_data.drop(columns = ['label'])
cv_number = 0
for train_index, test_index in skf.split(X,y):
    cv_number +=1
    model_filename = rs_folder+r'\cv' + str(cv_number) + '_model.sav'
    # load trained risk score model
    lambdas = pickle.load(open(model_filename, 'rb'))
    # get train/test data
    X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
    y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]
    data_test = y_test.merge(X_test,left_index=True, right_index=True)
    # get risk scores from model
    data_test, risk_prob_test = getScores(data_test,lambdas,thresh)
    # create new dataframe if first fold, else append to df, to collect all test results
    if cv_number == 1:
        riskscore_test = data_test
    else:
        riskscore_test = riskscore_test.append(data_test)
    # get metrics
    test_auc = roc_auc_score(data_test['label'],data_test['risk_pred'])
    test_conf = confusion_matrix(data_test['label'],data_test['prediction'])
    test_prec = precision_score(data_test['label'],data_test['prediction'])
    test_rec = recall_score(data_test['label'],data_test['prediction'])
    test_f1 = f1_score(data_test['label'],data_test['prediction'])
    # append resulsts
    results_rs = results_rs.append(pd.DataFrame(data={'cv_number':'CV'+str(cv_number)+'_test ',\
            'TN':test_conf[0,0],'FP':test_conf[0,1],'FN':test_conf[1,0],'TP':test_conf[1,1],\
                'precision':test_prec, 'recall':test_rec, 'f1_score':test_f1, 'auc':[test_auc] }))
    # plot AUC for test data
    fpr, tpr, _ = roc_curve(data_test['label'],data_test['risk_pred'])
    auc_table = auc_table.append({'cv_number':'CV'+str(cv_number)+' test','fpr':fpr,'tpr':tpr, 
                                        'auc':test_auc}, ignore_index=True)
    data_test['prediction'] = data_test['prediction'].astype('int')
auc_table.set_index('cv_number', inplace=True)
auc_range = results_rs['auc'].max() - results_rs['auc'].min()
results_rs = results_rs.append(pd.DataFrame(data={'cv_number':'Risk Score',\
            'auc':[results_rs.mean()['auc']], 'TN':results_rs.mean()['TN'],'FP':results_rs.mean()['FP'],\
                'FN':results_rs.mean()['FN'],'TP':results_rs.mean()['TP'], 'precision':results_rs.mean()['precision'],\
                    'recall':results_rs.mean()['recall'], 'f1_score':results_rs.mean()['f1_score'] }))
results_rs.loc[results_rs['cv_number']=='Risk Score','AUC Range'] = auc_range

# full data, load trained model and get scores
lambdas_full = pickle.load(open(rs_folder+r'\full_data_model.sav', 'rb'))
full_data_output, risk_prob_full = getScores(full_data,lambdas_full,thresh)
# don't need this file but it shows counts for each score and the corresponding probabilities
#risk_prob_full.to_csv(rs_folder+r'\risk_probs_full_model.csv')

#%% plot AUC vs baselines
# get avg CV metrics
mean_fpr = np.linspace(0, 1, 100)
auc_table_cv = auc_table#.loc[auc_table.index!='full data train']
# interpolate auc over range of 1 to 100
auc_table_cv['interp_tpr'] = auc_table_cv.apply(lambda row: np.interp(mean_fpr, row['fpr'],row['tpr']),axis=1)
risk_tprs_interp_mean = np.mean(auc_table_cv['interp_tpr'])
risk_auc_mean = auc_table_cv['auc'].mean()

# get decision tree metrics
results_dt = pd.DataFrame(columns = ['cv_number','TN','FP','FN','TP','precision','recall','f1_score','auc'])
dt_auc_table = pd.DataFrame(columns=['fold','fpr','tpr','auc'])
for i in range(5):
    probs = dt_results.loc[dt_results['fold']==i,'predicted_probability']
    preds = dt_results.loc[dt_results['fold']==i,'test_pred_labels']
    labels = dt_results.loc[dt_results['fold']==i, 'test_true_labels']
    auc = roc_auc_score(labels,probs)
    test_conf = confusion_matrix(labels,preds)
    test_prec = precision_score(labels,preds)
    test_rec = recall_score(labels,preds)
    test_f1 = f1_score(labels,preds)
    fpr, tpr, _ = roc_curve(labels,probs)
    dt_auc_table = dt_auc_table.append(pd.DataFrame(data={'fold':i,
            'fpr':[fpr],'tpr':[tpr],'auc':[auc]}))
    results_dt = results_dt.append(pd.DataFrame(data={'cv_number':str(i),\
            'auc':[auc],'TN':test_conf[0,0],'FP':test_conf[0,1],\
                'FN':test_conf[1,0],'TP':test_conf[1,1], 'precision':test_prec, 'recall':test_rec, 'f1_score':test_f1 }))
auc_range = results_dt['auc'].max() - results_dt['auc'].min()
results_dt = results_dt.append(pd.DataFrame(data={'cv_number':'Decision Tree',\
            'auc':[results_dt.mean()['auc']], 'TN':results_dt.mean()['TN'],'FP':results_dt.mean()['FP'],\
                'FN':results_dt.mean()['FN'],'TP':results_dt.mean()['TP'], 'precision':results_dt.mean()['precision'],\
                    'recall':results_dt.mean()['recall'], 'f1_score':results_dt.mean()['f1_score'] }))
results_dt.loc[results_dt['cv_number']=='Decision Tree','AUC Range'] = auc_range


dt_auc_table['interp_tpr'] = dt_auc_table.apply(lambda row: np.interp(mean_fpr, row['fpr'],row['tpr']),axis=1)
dt_tprs_interp_mean = np.mean(dt_auc_table['interp_tpr'])

dt_auc_mean = dt_auc_table['auc'].mean()

# get baseline metrics
baseline, baseline_agg = baselines(full_data, skf)
baseline_agg['method'] = ['SVM:                  ',
                         'Logistic Regression:  ',
                         'Naive Bayes:          ',
                         'Random Forest:        ']
baseline_agg = baseline_agg.set_index('method')

# plot baseline
fig = plt.figure(figsize=(8,6))
style = ['solid', 'dashdot','dotted','dashed']
alphas = [0.15,0.3,0.45,0.6]
j = -1
for i in baseline_agg.index:
    j += 1
    plt.plot(mean_fpr, 
              baseline_agg.loc[i]['interp_tpr'], 
              label="{} AUC={:.3f}".format(i, baseline_agg.loc[i]['auc']),
              alpha = alphas[j],
              color = 'black',
              linestyle = style[j],
              )
    
# plot risk score
plt.plot(mean_fpr,risk_tprs_interp_mean,
         label="{} AUC={:.3f}".format('Risk Score:           ', risk_auc_mean),
         alpha = 1, color ='black',linestyle='solid')

# plot decision trees
plt.plot(mean_fpr,dt_tprs_interp_mean,
          label="{} AUC={:.3f}".format('Optimal Decision Tree:', dt_auc_mean),
          alpha = 1, color ='black',linestyle='dashdot')


plt.plot([0,1], [0,1], color='gray', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('Average Test AUC over 5 Folds', fontweight='bold', fontsize=15)
plt.legend(prop={'size':12,'family':'monospace'}, loc='lower right')

os.mkdir('figures')
plt.savefig(r'figures\Figure4.pdf',dpi=1000,bbox_inches='tight')
plt.show()


#%% add average metrics to baseline table for Table 5 in manuscript
output_results = baseline_agg.reset_index().append(results_rs.loc[results_rs['cv_number']=='Risk Score'].rename(columns={'cv_number':'method'}))\
    .append(results_dt.loc[results_dt['cv_number']=='Decision Tree'].rename(columns={'cv_number':'method'})).drop(columns = ['interp_tpr'])
output_results.to_csv(r'figures\Table5.csv',index=False)

#%% get scores for full (unsampled data)
model_filename = r'riskscore_results\full_data_model.sav'
lambdas = pickle.load(open(model_filename, 'rb'))
presamp, risk_prob_presamp = getScores(presamp,lambdas,thresh)

#%% histogram of scores for full data
scores_with_intercept = risk_prob_full['score_with_intercept'].reset_index(drop=True).astype('int')
risk_prob_full['count_1'] = risk_prob_full['count']*risk_prob_full['risk_obs']

counts = risk_prob_full['count'].reset_index(drop=True)
counts_1 = risk_prob_full['count_1'].reset_index(drop=True).astype(int)

# risk pred and obs
risk_pred = risk_prob_full['risk_pred'].reset_index(drop=True).to_list()
risk_pred =  [f"{num:.3f}" for num in risk_pred]
risk_obs = risk_prob_full['risk_obs'].reset_index(drop=True).to_list()
risk_obs =  [f"{num:.3f}" for num in risk_obs]

#unsamp counts and risk obs
counts_presamp = risk_prob_presamp['count'].to_list()
risk_obs_presamp = risk_prob_presamp['risk_obs'].to_list()
risk_obs_presamp =  [f"{num:.3f}" for num in risk_obs_presamp]

fig = plt.figure(figsize=(8,5))
plt.bar(scores_with_intercept, counts,  width=1, label = "Score Count in\nUndersampled Data",edgecolor='black',color='black',alpha=0.4)
plt.bar(scores_with_intercept, counts_1, width=1, label = "Label 1 Score Count" ,edgecolor='black',color='black',alpha=0.6)
plt.tick_params(axis='x',bottom=False,labelbottom=False)
plt.ylabel("Count", fontsize=15)
plt.title('Risk Score: Count of Each Score by Label', fontweight='bold', fontsize=15)
plt.legend(prop={'size':12})

for i in range(len(scores_with_intercept)):
    if counts[i]-counts_1[i]>20:
        plt.text(scores_with_intercept[i],counts[i]+5,str(counts[i]),ha='center')
        plt.text(scores_with_intercept[i],counts_1[i]+5,'('+str(counts_1[i])+')',ha='center')
    else:
        plt.text(scores_with_intercept[i]+0.07,counts[i]+5,str(counts[i])+' ('+str(counts_1[i])+')',ha='center')

gap = 1/(2*(len(scores_with_intercept)+1))
plot_data = [scores_with_intercept.to_list(),
        risk_pred,
        risk_obs,
        counts_presamp,
        risk_obs_presamp]
plt.table(cellText=plot_data,
          rowLabels=['Score ', 'Predicted Risk ', 'Observed Risk in\nUndersampled Data',\
                     'Score Count in\nFull Data',\
                    'Observed Risk in\nFull Data'],
          bbox=[gap,-.46, 1-2*gap,.45],
          cellLoc='center'
          )
plt.savefig(r'figures\Figure2.pdf',dpi=1000, bbox_inches = "tight")    
plt.show()

#%% venn diagram
dt_results = dt_results.rename(columns = {'predicted_probability':'dt_prob','test_pred_labels':'dt_prediction'})
riskscore_test = riskscore_test.rename(columns = {'risk_pred':'risk_prob','prediction':'risk_prediction'})
riskscore_test['test_index'] = riskscore_test.index

results = dt_results.merge(riskscore_test, on = 'test_index')

# get counts for venn diagram
def count(risk, tree, label):
    subset = results.loc[results['risk_prediction']==risk]
    subset = subset.loc[results['dt_prediction']==tree]
    subset = subset.loc[results['label']==label]
    return len(subset)

just_risk = count(1,0,0)
just_tree = count(0,1,0)
risk_tree = count(1,1,0)
just_label = count(0,0,1)
risk_label = count(1,0,1)
tree_label = count(0,1,1)
risk_tree_label = count(1,1,1)

total_risk = just_risk+risk_tree+risk_label+risk_tree_label
total_tree = just_tree+risk_tree+tree_label+risk_tree_label
total_label = just_label+risk_label+tree_label+risk_tree_label

# create venn diagram
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(7,4))
venn3(subsets = (just_risk,just_tree,risk_tree,just_label,\
                  risk_label,tree_label,risk_tree_label), \
                  set_labels = ('Risk score\npredicted IMBs ('+str(total_risk)+')', 'Optimal decision tree\npredicted IMBs ('+str(total_tree)+')', 'Labeled as IMB (label = 1) ('+str(total_label)+')'), \
                alpha = 0)
venn3_circles(subsets = (just_risk,just_tree,risk_tree,just_label,\
                  risk_label,tree_label,risk_tree_label),linewidth=.5)

plt.savefig(r'figures/Figure5.pdf',dpi=1000,bbox_inches='tight')
plt.show()






