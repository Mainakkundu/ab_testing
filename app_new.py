import pandas as pd
import numpy as np
import streamlit as st
import datetime
import datetime as dt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
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
np.random.seed(8)
import utility

### Data for Normal analysis 
cup_df = pd.read_csv('./main/streamlit/data/cuped_data.csv') 
## Data for Causal Inferene 
np.random.seed(12345)
ar = np.r_[1, 0.9]
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)
X = 100 + arma_process.generate_sample(nsample=100)
y = 1.2 * X + np.random.normal(size=100)
y[70:] += 5
pre_post_data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X']) ###3
pre_period = [0, 69]
post_period = [70, 99]
st.title("""AB-Testing Tool """)

print('======================================================')
print('----------- Sample Size Estimation--------------------')
print('======================================================')
MENU = ['Sample-Size-Estimation','Stat Base Measurement','Analysis & Recommendation']
choice = st.sidebar.radio(''' Click here ''', MENU)
if choice == 'Sample-Size-Estimation':
    mean_sales = st.sidebar.number_input('Base-Mean',1)
    std_sales = st.sidebar.number_input('Base-StdDev',1)
    expected_lift = st.sidebar.number_input('Expected-Lift',1)
    alpha = st.sidebar.number_input('Alpha_Value',0.05)
    power = st.sidebar.number_input('Power_Value',0.8)
    avg_footfall_per_day = st.sidebar.number_input('Avg_foot_fall_per_day',100)
    pre_exp = utility.PreExpirement(alpha=alpha,power=power)
    sample_size = pre_exp.sample_size_calculator(MU_BASE=mean_sales,EXPECTED_LIFT =expected_lift,STD_DEV=std_sales)
    st.text(f"Sample Sizes Estimate as per choosen parameters is {sample_size} customers" )
    #st.subheader(sample_size)
    no_of_days = int(pre_exp.duration_calculation(sample_size,TRAFFIC_PER_DAY=avg_footfall_per_day))
    st.text(f"Need to run the test atleast {no_of_days} days , to attain seasonality")
    #st.subheader(sample_size)
    


