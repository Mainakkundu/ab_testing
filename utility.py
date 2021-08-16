from statsmodels.tsa.arima_process import ArmaProcess
from causalimpact import CausalImpact
from scipy import stats
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
import statistics
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 20,10
rcParams['font.size'] = 30
sns.set()
#np.random.seed(8)


## Global Variables defined by Users ##
MU_BASE = 40
STD_DEV=20
EXPECTED_LIFT=0.10


class PreExpirement():
    def __init__(self,df,power,alpha):
        self.df = df 
        self.power = power
        self.alpha = alpha
    
    def sample_size_calculator(self):
        from math import sqrt
        from statsmodels.stats.power import TTestIndPower
        mean0 = MU_BASE
        mean1 = MU_BASE +(MU_BASE*EXPECTED_LIFT)
        std = STD_DEV
        ## cohens-d is the effect size 
        cohens_d = (mean0 - mean1) / (sqrt((std ** 2 + std ** 2) / 2))

        effect = cohens_d
        alpha = 0.05
        power = 0.8

        analysis = TTestIndPower()
        #print(power)
        sample_size=analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
        sample_size = int(sample_size)
        return sample_size
    
    def duration_calculation(self,TRAFFIC_PER_DAY=100):
        sample_size=self.sample_size_calculator()
        no_of_days = sample_size/TRAFFIC_PER_DAY
        if no_of_days <= 7:
            no_of_days = 7
        else:
            no_of_days
        
        return no_of_days


def r2(self,x, y):
    '''
    measures the R2(correlation between 2 distribution)
    '''
    return stats.pearsonr(x, y)[0] ** 2



class StatsEngine():


    @staticmethod
    def r2(x, y):
        '''
        measures the R2(correlation between 2 distribution)
        '''
        return stats.pearsonr(x, y)[0] ** 2

    @staticmethod
    def t_distribution_ci(df,test_flag,control,test,metric,period):
        '''
        t-test measurement, with CI and p-value  
        '''

        signi = []
        p_value = []
        
        new_metric = period+'_'+metric
        test_data_A = df[df[test_flag]==control]
        test_data_B = df[df[test_flag]==test]
        test_data_A[new_metric] = test_data_A[new_metric].astype('float')
        test_data_B[new_metric] = test_data_B[new_metric].astype('float')
        print(test_data_A[new_metric].quantile(.995))
        #test_data_A_clean = test_data_A[(test_data_A[metric]>0) & (test_data_A[metric]<test_data_A[metric].quantile(.995))]
        test_data_A_clean = test_data_A
        print(test_data_B[new_metric].quantile(.995))
        #test_data_B_clean = test_data_B[(test_data_B[metric]>0) & (test_data_B[metric]<test_data_B[metric].quantile(.995))]
        test_data_B_clean = test_data_B
        #Combine the cleaned data sets as one
        test_data_clean = test_data_A_clean.append(test_data_B_clean)
        #Summarize the metrics:- Calculating totals
        test_summary1 = test_data_clean.groupby(test_flag).agg({
            new_metric:'sum'

        })
        #Summarize the metrics:- Calculating means
        test_summary2 = test_data_clean.groupby(test_flag).agg({
                    new_metric:'mean'
        })
        #Transposing the summaries
        test_summary1 = test_summary1.T
        test_summary2 = test_summary2.T

        #Initialize a dataframe with test stats
        test_stats = pd.DataFrame(columns = ['pct_lft','conf_int_lb','conf_int_ub','p-value'])
        #Concatenate the test stats with both the summaries
        test_summary1 = pd.concat([test_summary1,test_stats],axis=1,ignore_index=False,sort=False)
        #Calculate pct_lift for all the metrics
        test_summary1['pct_lft'] = (test_summary1[test]-test_summary1[control])/test_summary1[control]*100
        test_summary2 = pd.concat([test_summary2,test_stats],axis=1,ignore_index=False,sort=False)
        #Calculate pct_lift for all the metrics
        test_summary2['pct_lft'] = (test_summary2[test]-test_summary2[control])/test_summary2[control]*100

        cm = sms.CompareMeans(sms.DescrStatsW(test_data_A_clean[new_metric][test_data_A_clean[new_metric].notnull()]),
                    sms.DescrStatsW(test_data_B_clean[new_metric][test_data_B_clean[new_metric].notnull()]))
        lb,rb = cm.tconfint_diff(usevar='unequal',alternative='two-sided',alpha = 0.10)

        test_summary2['conf_int_lb'] = (rb*-1)/test_data_A_clean[new_metric].mean()
        test_summary2['conf_int_ub']=  (lb*-1)/test_data_A_clean[new_metric].mean()

        t_stat,test_summary2['p-value'] = sc.ttest_ind(test_data_A_clean[new_metric][test_data_A_clean[new_metric].notnull()],
                                    test_data_B_clean[new_metric][test_data_B_clean[new_metric].notnull()],equal_var = False)



        if (test_summary2['p-value'].iloc[0] < alpha) and (test_summary2['pct_lft'].iloc[0]  > 0):
            signi.append('Significant with lift')
        elif (test_summary2['p-value'].iloc[0] < alpha) and (test_summary2['pct_lft'].iloc[0] < 0):
            signi.append('Significanct ,control performance better than test')
        elif (test_summary2['p-value'].iloc[0] > alpha) and (test_summary2['pct_lft'].iloc[0] < 0):
            signi.append('Not significanct with negative lift')
        elif(test_summary2['p-value'].iloc[0] > alpha) and (test_summary2['pct_lft'].iloc[0] > 0):
            signi.append('Not significant with positive lift')
        else:
            signi.append('Nothing')

        print(signi)

        test_summary2['sigificance'] = signi
        return test_summary2