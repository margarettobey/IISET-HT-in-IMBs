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

hyperparameters = {
    "regularization": 0.000001,
    "time_limit": 18000,
    "verbose": True,
    "objective": "wacc",
    "w":4
}

model = GOSDT(hyperparameters)
model.fit(X, y)
print("Execution Time: {}".format(model.time))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print("Size: {}".format(model.leaves()))
print("Loss: {}".format(1 - training_accuracy))
print("Risk: {}".format(
    model.leaves() * hyperparameters["regularization"]
    + 1 - training_accuracy))
#model.tree.__initialize_training_loss__(X, y)
print(model.tree)
print(model.latex())

print(model.tree.features())

print(classification_report(y, prediction, digits=4))
print(confusion_matrix(y,prediction))
