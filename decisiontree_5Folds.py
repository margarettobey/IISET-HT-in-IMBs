import pandas as pd
import numpy as np
from model.gosdt import GOSDT

from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, f1_score    
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

df_allData = pd.read_csv("selected_features_1612and403.csv", index_col=[0])

# Select Yelp only variables
allYelpAndSelected_column = \
 ['yelp_massageCat', 'yelp_spaCat', 'yelp_category_reflexology',\
  'yelp_reviewRating_min_NEW_is5', 'yelp_average_all_ratings_NEW_moreThan4',\
  'yelp_phone_advertisement', 'yelp_business_name_NEW_Combine',\
  'yelp_reviewRating_std_NEW_high_std',\
  'yelp_reviewRating_std_NEW_low_std',\
  'yelp_reviewRating_std_NEW_zero_std',\
  'yelp_revCount_NEW_0to5', 'yelp_revCount_NEW_moreThan20',\
  'yelp_lexicon_score_mean_NEW_high_lexiconmean',\
  'yelp_lexicon_score_mean_NEW_low_lexiconmean',\
  'yelp_lexicon_score_mean_NEW_zero_lexiconmean',\
  'yelp_authorGender_PctMale_NEW_high_pctmale',\
  'yelp_authorGender_PctMale_NEW_low_pctmale',\
  'census_pct_nonwhite_NEW_low',\
  'census_pct_nonwhite_NEW_high',\
  'census_pct_foreign_born_NEW_low',\
  'census_pct_households_with_children_NEW_low',\
  'census_pct_20_to_29_NEW_low',\
  'min_dist_base_NEW_long']


X = df_allData[allYelpAndSelected_column]
y = df_allData[df_allData.columns[0]].to_frame()

xlwriter = pd.ExcelWriter('df_5Folds_output.xlsx')

regularization_list = [0.000001]
weight_list = [4]

count=1
for reg in regularization_list:
    for wt in weight_list:
        print(count)
        E1_Five_TestA, E1_Five_Recall, E1_Five_Precision, E1_Five_F1, E1_Five_AUC = [],[],[],[],[]
        E1_test_pred, E1_test_pred_prob, E1_test_labels = [],[],[]
        E1_optimalTime, E1_currentOptimal, E1_treeFeatures = [],[],[]

        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            hyperparameters = {
            "regularization": reg,
            "time_limit":5400,
            "verbose": False,
            "objective": "wacc",
            "w":wt
            }

            model = GOSDT(hyperparameters)
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)
            prediction_confidence = model.tree.confidence(X_test)
            prediction_prob = []
            for idx in range(len(prediction_confidence)):
                if prediction[idx] == 0:
                    prediction_prob.append(1-prediction_confidence[idx])
                else:
                    prediction_prob.append(prediction_confidence[idx])
            E1_test_pred.append(prediction)
            E1_test_pred_prob.append(prediction_prob)
            E1_test_labels.append(y_test['label'].ravel())
            
            E1_optimalTime.append(model.optimalTime)
            E1_currentOptimal.append(model.currentOptimal)
           
            idxxSet = set()
            nodes = [model.tree.source]
            while len(nodes) > 0:
                   node = nodes.pop()
                   if "prediction" in node:
                       continue
                   else:
                       idxxSet.add(node["feature"])
                       nodes.append(node["true"])
                       nodes.append(node["false"])
            E1_treeFeatures.append([X.columns[idxx] for idxx in idxxSet])

        # Save Results:
        result_dict = {}
        for i in range(5):
            #print("Fold"+str(i)print(classification_report(E1_test_labels[i],E1_test_pred[i],digits=4))
            confusion_M = confusion_matrix(E1_test_labels[i],E1_test_pred[i])
            TN = confusion_M[0,0]
            FP = confusion_M[0,1]
            FN = confusion_M[1,0]
            TP = confusion_M[1,1]
            #print(confusion_matrix(E1_test_labels[i],E1_test_pred[i]))
            fold_classification_report_dict = classification_report(E1_test_labels[i],\
                                                                    E1_test_pred[i],digits=4,output_dict=True)
            testing_accuracy = accuracy_score(E1_test_labels[i],E1_test_pred[i])
            testing_recall = fold_classification_report_dict['1']['recall']
            testing_precision = fold_classification_report_dict['1']['precision']
            testing_f1 = fold_classification_report_dict['1']['f1-score']
            testing_AUC = roc_auc_score(E1_test_labels[i],E1_test_pred_prob[i])
            #print("Testing Accuracy:", testing_accuracy)
            #print("Testing Recall:", testing_recall)
            #print("Testing Precision:", testing_precision)
            #print("Testing F1-score:", testing_f1)
            #print("Testing AUC:", testing_AUC)

            value_list = [TN,FP,FN,TP,testing_accuracy,testing_recall,testing_precision,testing_f1,testing_AUC,\
                          E1_optimalTime[i], E1_currentOptimal[i], E1_treeFeatures[i]]
            result_dict["Fold{0}".format(i)] = value_list

            E1_Five_TestA.append(testing_accuracy)
            E1_Five_Recall.append(testing_recall)
            E1_Five_Precision.append(testing_precision)
            E1_Five_F1.append(testing_f1)
            E1_Five_AUC.append(testing_AUC)

        result_df = pd.DataFrame(result_dict)

        confusion_M_ALL = confusion_matrix(np.ravel(E1_test_labels),np.ravel(E1_test_pred))
        TN_all = confusion_M_ALL[0,0]
        FP_all = confusion_M_ALL[0,1]
        FN_all = confusion_M_ALL[1,0]
        TP_all = confusion_M_ALL[1,1]  
        average_value_list = [TN_all,FP_all,FN_all,TP_all,np.mean(E1_Five_TestA),np.mean(E1_Five_Recall),\
                              np.mean(E1_Five_Precision), np.mean(E1_Five_F1), np.mean(E1_Five_AUC),\
                              None, None, None]
        result_df['Average'] = average_value_list
        
        result_df.index = ['TN','FP','FN','TP','Accuracy','Recall','Precision','F1-Score','AUC',\
                           'optimalTime', 'currentOptimal', 'treeFeatures']
        
        result_df.to_excel(xlwriter, sheet_name = str(reg)+"_"+str(wt), index = True)
        count = count + 1      
xlwriter.close()
