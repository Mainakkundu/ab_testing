print('--------Imports --------')
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import datetime as dt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
import os

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pylab import rcParams
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
import textwrap
from numpy import nansum
from numpy import nanmean
import pandas as pd
import statsmodels.stats.api as sms
from scipy import stats as sc
from causalimpact import CausalImpact
from statsmodels.formula.api import ols
from PIL import Image
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 20,10
rcParams['font.size'] = 30
sns.set()
np.random.seed(8)
print('----------------------------')

os.chdir('/Users/maikundu/Desktop/My_Learning/MediumBlog/AB_test_app')

print('===============================================')
print('----------- Utility Functions -----------------')
print('===============================================')

def min_detectable_data_prep(base_mean,base_std,min_detectable_change):
  '''
  >>> Prepare the data to incorporate the min-detectable changes which will
  pass into sample_size_calculator
  '''
  mu_final_l = []
  df_f = pd.DataFrame()
  for i in min_detectable_change:
    mu = base_mean+(base_mean*i)
    mu_final_l.append(mu)
    fd = pd.DataFrame()
    #fd['business_metric'] = [business_metric]
    fd['mu_base'] = [base_mean]
    fd['std_base'] =[base_std]
    fd['detectable_effect'] = [i]
    fd['mu_hat'] = [mu]
    df_f = df_f.append(fd)
  return (df_f)

def sample_size_calculator(mu_base,mu_hat,std_base,alpha=0.05,power=0.8):
    '''
    >>> Sample size calculation for Hypothesis Testing
    '''
    from math import sqrt
    from statsmodels.stats.power import TTestIndPower
    mean0 = mu_base
    mean1 = mu_hat
    std = std_base

    cohens_d = (mean0 - mean1) / (sqrt((std ** 2 + std ** 2) / 2))

    effect = cohens_d
    #alpha = 0.05
    #power = 0.8

    analysis = TTestIndPower()
    sample_size=analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    return(int(sample_size))

def week_flg(pos_busn_dt,pre_prd_start='2020-07-06',pre_prd_end='2020-10-26',post_prd_strt='2020-11-02',post_prd_end='2020-11-29'):
    '''
    >>> Take input from streamlit app dates for pre & post
    '''
    if (pre_prd_end >= str(pos_busn_dt) >= pre_prd_start):
        return 'pre'
    elif(post_prd_end >= str(pos_busn_dt) >= post_prd_strt):
        return 'post'

def metric_data_prep(df_merg,KPI_original='tot_bask_spend',primary_key='card_code',KPI_rename='sales'):
    '''
    >>> Prepare the Base metric data
    '''
    df_merg['week_flg'] = np.vectorize(week_flg)(df_merg['pos_busn_dt'])
    df_merg['week_flg'] = df_merg['week_flg'].fillna(0)
    g = df.groupby([primary_key,'week_flg']).agg({KPI:['sum']})
    g.rename(columns = {KPI_original:KPI_rename}, inplace = True)
    g_pvt = pd.pivot(g,columns='week_flg',values=KPI_rename,index=primary_key)
    g_pvt =g_pvt.fillna(0)
    g_pvt = pd.DataFrame(g_pvt)
    g_pvt['post_'+KPI_rename] =  g_pvt['post']
    g_pvt['pre_'+KPI_rename] = g_pvt['pre']
    del g_pvt['post']
    del g_pvt['pre']
    del g_pvt[0]
    return g_pvt



def r2(x, y):
    '''
    >>> Stronger the relationship high the R-square
    '''
    return stats.pearsonr(x, y)[0] ** 2

def CUPED(df,KPI):
    '''
    >>> Microsoft's classical approcah to reduce the variance and increase the power
    'https://frankhopkins.github.io/py_ab_fh/5_Variance_Reduction_Methods.html'
    '''
    pre_experiment_KPI = KPI+'_pre_experiment'
    covariance = np.cov(df[KPI], df[pre_experiment_KPI])
    variance = np.cov(df[pre_experiment_KPI])
    theta_calc = covariance / variance
    theta_calc_reshape = theta_calc.reshape(4,1)
    theta = theta_calc_reshape[1]
    print(theta)
    df['CUPED-adjusted_metric'] = df[KPI] - (df[pre_experiment_KPI] - statistics.mean(df[pre_experiment_KPI])) * theta

    std_pvs = statistics.stdev(df[KPI])
    std_CUPED = statistics.stdev(df['CUPED-adjusted_metric'])
    mean_pvs = statistics.mean(df[KPI])
    mean_CUPED = statistics.mean(df['CUPED-adjusted_metric'])

    relative_pvs = std_pvs / mean_pvs
    relative_cuped = std_CUPED / mean_CUPED
    relative_diff = (relative_cuped - relative_pvs) / relative_pvs

    print("The mean of the KPI  is %s."
    % round(mean_pvs,4),
    "The mean of the CUPED-adjusted KPI is % s."
    % round(mean_CUPED,4))


    print ("The standard deviation of KPI is % s."
        % round(std_pvs,4),
          "The standard deviation of the CUPED-adjusted KPI is % s."
           % round(std_CUPED,4))

    print("The relative reduction in standard deviation was % s"
        % round(relative_diff*100,5),"%",)

    return df








print('===============================================')
print('----------- Application Run--------------------')
print('===============================================')

st.title("""
         AB-Testing Tool """)
html_temp = """
<div style="background-color:orange;padding:10px">
<h2 style="color:black;text-align:center;">Offline Marketing Campaigns</h2>
</div>
"""
