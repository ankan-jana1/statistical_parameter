# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:45:38 2021

@author: ankan_jana
"""


import numpy as np
import pandas as pd
import logging
logging.basicConfig(format='%(levelname)s: %(module)s.%(funcName)s(): %(message)s')


def pbias(evaluation, simulation):
    """
    Procentual Bias
        .. math::
         PBias= 100 * \\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})}{\\sum_{i=1}^{N}(e_{i})}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: PBias
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        sim = np.array(simulation)
        obs = np.array(evaluation)
        return 100 * (float(np.nansum(sim - obs)) / float(np.nansum(obs)))

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan
def correlationcoefficient(evaluation, simulation):
    """
    Correlation Coefficient
        .. math::
         r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Corelation Coefficient
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        correlation_coefficient = np.corrcoef(evaluation, simulation)[0, 1]
        return correlation_coefficient
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

def rsquared(evaluation, simulation):
    """
    Coefficient of Determination
        .. math::
         r^2=(\\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})^2
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Coefficient of Determination
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        return correlationcoefficient(evaluation, simulation)**2
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan
def mse(evaluation, simulation):
    """
    Mean Squared Error
        .. math::
         MSE=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Mean Squared Error
    :rtype: float
    """

    if len(evaluation) == len(simulation):
        obs, sim = np.array(evaluation), np.array(simulation)
        mse = np.nanmean((obs - sim)**2)
        return mse
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


def rmse(evaluation, simulation):
    """
    Root Mean Squared Error
        .. math::
         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Root Mean Squared Error
    :rtype: float
    """
    if len(evaluation) == len(simulation) > 0:
        return np.sqrt(mse(evaluation, simulation))
    else:
        logging.warning("evaluation and simulation lists do not have the same length.")
        return np.nan

_all_functions = [correlationcoefficient, rsquared, nashsutcliffe, pbias, rmse]
                  

def calculate_all_functions(evaluation, simulation):
    """
    Calculates all objective functions from spotpy.objectivefunctions
    and returns the results as a list of name/value pairs
    :param evaluation: a sequence of evaluation data
    :param simulation: a sequence of simulation data
    :return: A list of (name, value) tuples
    """

    result = []
    for f in _all_functions:
        # Check if the name is not private and attr is a function but not this

        try:
            result.append((f.__name__, f(evaluation, simulation)))
        except:
            result.append((f.__name__, np.nan))

    return result

data=pd.read_excel('newdelhi.xlsx')
output=('outnewdelhi.xlsx')
data_imd = list(data['IMD'])
data_sebal= list(data['SEBAL'])
data_sebs = list(data['SEBS'])
data_ssebi = list(data['S-SEBI'])
data_ssebop = list(data['SSEBop'])
data_metric = list(data['METRIC'])

cal_sebal=calculate_all_functions(data_imd, data_sebal)
cal_sebal= pd.DataFrame (cal_sebal, columns = ['statistical parameters', 'sebal'])
cal_metric=calculate_all_functions(data_imd, data_metric)
cal_metric= pd.DataFrame (cal_metric, columns = ['statistical parameters', 'metric'])
cal_sebs=calculate_all_functions(data_imd, data_sebs)
cal_sebs= pd.DataFrame (cal_sebs, columns = ['statistical parameters', 'sebs'])
cal_ssebi=calculate_all_functions(data_imd, data_ssebi)
cal_ssebi= pd.DataFrame (cal_ssebi, columns = ['statistical parameters', 'ssebi'])
cal_ssebop=calculate_all_functions(data_imd, data_ssebop)
cal_ssebop= pd.DataFrame (cal_ssebop, columns = ['statistical parameters', 'ssebop'])
#df_3 = pd.concat([cal_sebal, cal_metric, cal_sebs, cal_ssebi, cal_ssebop])
#pd.merge(cal_sebal, cal_metric, cal_sebs, cal_ssebi, cal_ssebop, how='left', on=['statistical parameters'])
df_col1=pd.merge(cal_sebal, cal_metric, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True)
df_col2=pd.merge(cal_sebs, cal_ssebi, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True)
df_col3=pd.merge(df_col1, df_col2, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True)
df_col=pd.merge(df_col3, cal_ssebop, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True)

df_col.to_excel(output)
