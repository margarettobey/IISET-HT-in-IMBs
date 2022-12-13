# IISET-HT-in-IMBs

This material will be available at: https://doi.org/10.5281/zenodo.7430460 

This is the code for the IISE Transactions article "Interpretable Models for the Automated Detection of Human Trafficking in Illicit Massage Businesses" by Tobey et al., 2022 (https://doi.org/10.1080/24725854.2022.2113187)

This code requires installation of the risk-slim package (https://github.com/ustunb/risk-slim) and GOSDT package (https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees)

Input data descriptions:

A. "selected_features_9070and403.csv" is the entire labeled data set after feature selection

B. "selected_features_1612and403.csv" is the undersampled version of A

C. "data_categorical_9070and403.csv" is used for the univariate and multivariate anlysis before feature selection

Steps:

1. Run "riskscores.py" to train the risk score model (5 CV folds plus 1 final run on full dataset B)
	inputs: "selected_features_1612and403.csv"
	outputs: creates folder "riskscore_results" and saves the 6 trained models

2. For running decision tree models, replace the "encode.py" file in the GOSDT package with the new "encode.py" file 
	(reason: read input data directly without discretization).

3. Run decisiontree_wholeData.py to train the decision tree model on full data set
	inputs: "selected_features_9070and403.csv"
	outputs: output in python console for drawing final tree model

4. Run "decisiontree_5Folds.py" for cross validation on decision tree models
	inputs: "selected_features_9070and403.csv"
	outputs: "dt_5Folds_output.xlsx" displays decicision tree computational results,
		"5FoldSCV_prediction_info.csv" contains predictions for each test set 

5. Run results_and_plots.py
	inputs: trained risk score models, decision tree predictions: "5FoldSCV_prediction_info.csv"
	outputs: manuscript table 5 and figures 2, 4, and 5
