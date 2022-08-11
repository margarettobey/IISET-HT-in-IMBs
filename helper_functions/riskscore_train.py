# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:47:50 2022

@author: mgtobey
"""
import pandas as pd
import os
import numpy as np
import riskslim
import math

#%% run risk slim
# data
def riskscore_train(data,size,runtime):
    # problem parameters
    max_coefficient = 5                                        # value of largest/smallest coefficient
    max_L0_value = size                                            # maximum model size (set as float(inf))
    max_offset = 50                                             # maximum value of offset parameter (optional)
    c0_value = 1e-6                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    
    os.chdir(r"C:\Users\mgtobey\risk-slim")
    
    # create coefficient set and set the value of the offset parameter
    coef_set = riskslim.CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0)
    coef_set.update_intercept_bounds(X = data['X'], y = data['Y'], max_offset = max_offset)
    
    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }
    
    # major settings (see riskslim_ex_02_complete for full set of options)
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        #
        # LCPA Settings
        'max_runtime': runtime,                               # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': True,                     # print CPLEX progress on screen
        'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')
        #
        # LCPA Improvements
        'round_flag': True,                                # round continuous solutions with SeqRd
        'polish_flag': True,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': True,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        #
        # Initialization
        'initialization_flag': True,                       # use initialization procedure
        'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,
        #
        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }
    
    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = riskslim.run_lattice_cpa(data, constraints, settings)
    
    return(model_info,mip_info, lcpa_info)

def getScores(data_in, lambdas, threshold):
    X_data = np.concatenate((np.ones((len(data_in),1)),np.array(data_in.drop(columns=['label']))),axis=1)
    data_out = data_in.copy()
    data_out['score_with_intercept'] = np.dot(X_data,lambdas)
    data_out['score'] = data_out['score_with_intercept'] - lambdas[0]
    data_out['risk_pred'] = data_out['score_with_intercept'].apply(lambda x: 1/(1+math.exp(-x)))
    data_out['prediction'] = data_out['risk_pred'].apply(lambda x: x>=threshold).astype('int')
    risk_prob = pd.DataFrame(columns=['score_with_intercept','score','risk_pred','risk_obs','count'])
    for score_int in data_out['score_with_intercept'].unique():
        score = score_int - lambdas[0]
        filtered = data_out.loc[data_out['score_with_intercept']==score_int]
        pred = 1/(1+math.exp(-score_int))
        obs = sum(filtered['label'])/len(filtered)
        count = len(filtered)
        risk_prob = risk_prob.append(pd.DataFrame(data={'score_with_intercept':[score_int],'score':[score],'risk_pred':[pred],'risk_obs':[obs],'count':count}))
    risk_prob = risk_prob.sort_values('risk_pred')
    return data_out, risk_prob

def calculate_cal(risk_prob):
    dif = abs(risk_prob['risk_pred']-risk_prob['risk_obs'])
    mult = dif*risk_prob['count']
    cal = sum(mult)/sum(risk_prob['count'])
    return cal
