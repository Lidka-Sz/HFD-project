# # PROJECT SCRIPT (Group 1 pair of assets)


# Common assumptions for strategies on group 1 assets:
# 
# -do not use in calculations the data from the first and last 10 minutes of the session (9:31-9:40 and 15:51-16:00) – put missing values there,
# -do not hold positions overnight (exit all positions 20 minutes before the session end, i.e. at 15:40),
# -do not trade within the first 25 minutes of stocks quotations (9:31-9:55), but DO use the data for 9:41-9:55 in calculations of signal, volatility


# Common assumptions for startegy building:
# 
# Within each of the above groups of assets you can
# -trade just a single asset, or
# -put (selected) assets together in pair(s) as spreads, or
# -trade each of selected assets separately and treat them as a portfolio (applying the same or different strategy for each asset).
# 
# If trading more than one asset (spread), remember to include positive transaction costs for each of them.


# ## DATA AND DEPENDECIES LOADING


# We load the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
import statsmodels.api as sm
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from arch.unitroot import ADF, PhillipsPerron, KPSS
from sklearn.metrics import confusion_matrix
import warnings

# we ignore deprecation warnings and futurewarnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # ignore future warnings
warnings.simplefilter(action="ignore", category=UserWarning)  # ignore warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)  # ignore runtime warnings

# lets add the functions path to sys.path
import sys
sys.path.append('functions')

# ## EXPLORATION OF STRATEGIES


# ### Q1 2023 DATA INSPECTION


#Aggregation of datasets
import pandas as pd

quarters = [
    '2023_Q1', '2023_Q3', '2023_Q4',
    '2024_Q2', '2024_Q4',
    '2025_Q1', '2025_Q2'
]

dfs = []

for quarter in quarters:
    print(f'Processing quarter: {quarter}')
    df = pd.read_parquet(f'data/data1_{quarter}.parquet')
    dfs.append(df)

# aggregate all quarters
data1 = pd.concat(dfs, axis=0)

# if datetime exists
data1['datetime'] = pd.to_datetime(data1['datetime'])
data1 = data1.sort_values('datetime').set_index('datetime')


data1.info()

data1.head()

# assumption 1
# do not use in calculations the data from the first and last 10 minutes 
# of the session (9:31-9:40 and 15:51-16:00) – put missing values there,

data1.loc[data1.between_time("9:31", "9:40").index] = np.nan
data1.loc[data1.between_time("15:51", "16:00").index] = np.nan


data1.describe()

print(data1.isna().sum())

data1.head(11) # 11 NA's at the beginning and the end of each day

data1.plot(y = 'NQ', use_index = True, figsize = (10, 5))
plt.title("Nasdaq futures quotations")   

data1.plot(y = 'SP', use_index = True, figsize = (10, 5))
plt.title("S&P500 futures quotations")   

#Saving dataset for further calculations
data1.to_pickle("data1_input.pkl")

# #### ROLLING WINDOW ANALYSES


# 1) Rolling volatility


# fill missing data with the last non-missing value
# (forward fill method)

data1.ffill(inplace = True)

print(data1.isna().sum())

data1.head(400) # only 'first' 399 NA's left

# based on closing prices
# we will calculate logarithmic rates of return for
# all series in basis points (bps) 1bps = 0.01% = 0.0001,
# to obtain basis points of return, multiply by 10,000

# Shift index by desired number of periods - can be positive or negative
data1_r = np.log(data1 / data1.shift(1)) * 10000

data1_r.head(401)

data1_r.columns = ["r_NQ", "r_SP"]


# Let's combine onlyprices and rates of return 

data1_roll = pd.concat(
    [data1[['NQ', 'SP']], 
     data1_r[['r_NQ', 'r_SP']]],
    axis=1
)

data1_roll.head(401)

# Let's see a figure of all columns
# with closing prices and yields for NQ and SP
# on sublots

data1_roll.plot(subplots = True,  
                  layout = (2, 2), 
                  title = "Quotations of NQ and SP",
                  figsize = (12, 10))
plt.show() #return plots: measured in bps

# Let's check the standard deviation of NQ's returns for the entire sample

NQ_r_std = data1_roll['r_NQ'].std()

print("Standard deviation of NQ returns:")
print(NQ_r_std)

# Let's check the standard deviation of SP's returns for the entire sample

SP_r_std = data1_roll['r_SP'].std()

print("Standard deviation of SP returns:")
print(SP_r_std) #lower

# let's calculate the standard deviation for
# rolling intervals (in a 60-minute window)
data1_roll['r_NQ_rolling_std'] = data1_roll['r_NQ'].rolling(window = 60).std(ddof=1)

# let's calculate the standard deviation for
# rolling intervals (in a 60-minute window)
data1_roll['r_SP_rolling_std'] = data1_roll['r_SP'].rolling(window = 60).std(ddof=1)

# Let's convert the datetime index to text and pass it as x-axis
data1_roll_plot = data1_roll.copy()
data1_roll_plot['time'] = data1_roll_plot.index.astype(str)

# We reset the index to make 'time' a column
data1_roll_plot = data1_roll_plot.reset_index(drop = True)

# plot the figure of 'r_NQ_rolling_std'

data1_roll_plot.plot(
    x = 'time',
    y = 'r_NQ_rolling_std',
    title = "Standard deviation of NQ returns",
    figsize = (12, 6)
)
# let's add a reference line at the standard deviation level
# SP_r_std
plt.axhline(y = SP_r_std, 
            color = 'r',
            linestyle = '--', 
            label = 'Standard deviation across the entire sample')
plt.legend()    

plt.show()
# volatility was lower than average for later phase of the period

# 2) Rolling correlation


# NQ and SP quotation price correlation in the entire sample

correlation_p = data1_roll['NQ'].corr(data1_roll['SP'])

print("NQ and SP closing price correlation:")
print(correlation_p) #very high positive correlation value!

# NQ and SP quotation price correlation in the entire sample

correlation_r = data1_roll['r_NQ'].corr(data1_roll['r_SP'])

print("NQ and SP rate of returns correlation:")
print(correlation_r)

# slightly lower than for the closing prices - which is typical

# rolling analysis of price correlations in a 2-hour (120-minute) timeframe

data1_roll['rollcorr120_NQ_SP'] = data1_roll['NQ'].rolling(window=120).corr(data1_roll['SP'])

data1_roll.head(401)

# Let's see how correlation changes over time

# Let's convert the datetime index to a text index and pass it as the x-axis
data1_roll_plot = data1_roll.copy()
data1_roll_plot['time'] = data1_roll_plot.index.astype(str)

# Reset the index so that 'time' is a column
data1_roll_plot = data1_roll_plot.reset_index(drop = True)

# Let's draw the graph
data1_roll_plot.plot(
    x = 'time',
    y = 'rollcorr120_NQ_SP',
    title = "Rolling correlation of NQ and SP prices (120 minutes)",
    figsize = (12, 6)
)

# let's add a reference line at the correlation level in the entire sample
plt.axhline(y = correlation_p,
            color = 'r',
            linestyle = '--',
            label = 'Correlation in the entire sample')

plt.legend()
plt.show()

# let's save the rolling window file to the pickle file
# for further analysis

data1.to_pickle("data1_input_filled_na.pkl")
data1_roll.to_pickle("data1_roll.pkl")

# 3) Rolling linear and quantile regression


data1_roll = pd.read_pickle("data1_roll.pkl")
data1.head(11)

# lets start with a scatter plot of returns of both stocks

plt.figure(figsize=(8,6))
plt.scatter(data1_roll["r_NQ"], data1_roll["r_SP"], alpha=0.5)
plt.title("Scatter plot of returns of NQ and SP futures (Q1 2023)")
plt.xlabel("NQ returns")
plt.ylabel("SP returns")
plt.grid()
plt.show()

# looks like there is positive relationship between returns of both stocks (strong linearity)

## ATTENTION!
data1_roll.head(401) # soem NA's in the dataset

# lets remove the first 400 rows  from the dataset
data1_roll = data1_roll.iloc[400:, :]

# let's quantify it with linear regression

X = data1_roll["r_NQ"]
y = data1_roll["r_SP"]
X = sm.add_constant(X)  # adding a constant term for intercept
model_ols = sm.OLS(y, X).fit()
print(model_ols.summary())

# relationship highly significant
# regression coefficient = 0.7117, R2 = 0.807
# constant term not significant

# lets add a regression line to the scatter plot

plt.figure(figsize=(8,6))
plt.scatter(data1_roll["r_NQ"], 
            data1_roll["r_SP"], 
            alpha=0.5,
            label="Data points")
# regression line
plt.plot(data1_roll["r_NQ"],
         model_ols.predict(X), 
         color='red', label="OLS regression line")
plt.title("Scatter plot of returns of NQ and SP with regression line")
plt.xlabel("NQ returns")
plt.ylabel("SP returns")
plt.legend()
plt.grid()
plt.show()

# lets estimate quantile regression at median (0.5 quantile)
model_qr_median = sm.QuantReg(y, X).fit(q = 0.5)
print(model_qr_median.summary())

# the impact of NQ returns on conditional median 
# of SP returns is statistically significant
# and the slope is slightly more steep (0.7171) than for OLS!

plt.figure(figsize=(8,6))

plt.scatter(
    data1_roll["r_NQ"],
    data1_roll["r_SP"],
    alpha=0.5,
    label="Data points"
)

# OLS regression line
plt.plot(
    data1_roll["r_NQ"],
    model_ols.predict(X),
    color="red",
    label="OLS regression line"
)

# Quantile regression line
plt.plot(
    data1_roll["r_NQ"],
    model_qr_median.predict(X),
    color="green",
    label="Quantile regression (median) line"
)

plt.title("Scatter plot of returns of NQ and SP with regression lines")
plt.xlabel("NQ returns")
plt.ylabel("SP returns")

plt.xlim(-30, 30)
plt.ylim(-30, 30)

plt.legend()
plt.grid()
plt.show()


# lets apply the Engle-Granger two-step method

# Step 1: estimate the long-run relationship with OLS
X_price = data1_roll["NQ"]
y_price = data1_roll["SP"]
X_price = sm.add_constant(X_price)  # adding a constant term for intercept

model_ols_price = sm.OLS(y_price, X_price).fit()
print(model_ols_price.summary())


# Step 2: obtain the residuals from the OLS regression
residuals = model_ols_price.resid

# and test them for stationarity with the ADF test
adf_test = ADF(residuals)

# by default number of lags is selected automatically based on AIC
# WHICH is not correct as it does not take into account potential autocorrelation
# and trend = 'c' (constant/drift) is used

print(adf_test.summary().as_text())

# We cannot reject H0 about NON-stationarity (but close to p-value=5% level)

# lets use simpler settings to speed up the calculations

adf_test2 = ADF(residuals,
                lags = 5, # how many augmentations
                # random walk with a constant/drift - default
                # (alternative: 'n', 'ct', 'ctt' - last with quadratic trend also)
                trend = 'c') 
print(adf_test2.summary().as_text())
# conclusion on cointegration is same as above, even closer

# lets apply a Phillips-Perron test

pp_test = PhillipsPerron(residuals) 
# by default lags is set automatically to 12 * (nobs/100) ** (1/4)
# and trend = 'c' (constant/drift) is used

print(pp_test.summary().as_text())
# same conclusion and in ADF test:
# we cannot reject H0 about NON-stationarity (but p-value very close to 5% level)

# (also the test statistic and its p-value are 
# very close to that from ADF test)

# and in the end the KPSS test

kpss_test = KPSS(residuals) 
# by default the number of lags is calculated with
# the data-dependent method of Hobijn et al. (1998) 
# and trend = 'c' (constant/drift) is used

print(kpss_test.summary().as_text())
# same conclusion as in two earlier tests:
# we reject H0 about stationarity!!!

# How to extract p-values from the tests?

print("p-value from ADF test:", adf_test2.pvalue)
print("p-value from Phillips-Perron test:", pp_test.pvalue)
print("p-value from KPSS test:", kpss_test.pvalue)

#Granger causality test
print("We check if SP prices Granger-cause NQ prices")
# second column is the causing variable
granger_test = grangercausalitytests(data1_roll[["NQ", "SP"]], 
                      # if we put maxlag = [10],
                      # we test only for lag = 10
                      maxlag = [10])
print("\n")
print("We check if NQ prices Granger-cause SP prices")
granger_test = grangercausalitytests(data1_roll[["SP", "NQ"]], 
                      maxlag = [10])

# however, with 10 minute horizon we have bidirectional Granger causality!
# (the null hypothesis of no causality is rejected at any significance level)

# 3a) Rolling  linear regression


# lets apply rolling OLS regression
# based on a moving window of 180 minutes
# The cleanest, loop-free way to do this 
# is to use the RollingOLS function from statsmodels

# Prepare data
y = data1_roll["r_SP"]
X = sm.add_constant(data1_roll["r_NQ"])

# Run rolling OLS
rolling_model = RollingOLS(y, X, window = 180, min_nobs = 180)
rolling_results = rolling_model.fit()

# unfortunately, there is no similar equivalent 
# for quantile regression in python...

# Extract rolling parameters
params = rolling_results.params
rsq = rolling_results.rsquared

# Combine results into a single DataFrame
rolling_out = pd.concat([params, rsq.rename("R_squared")], axis=1).dropna()

rolling_out.head()


# how to extract a beta coefficient from the model?
beta_ols = model_ols.params["r_NQ"]
print("Beta coefficient from OLS regression:", beta_ols)

# lets plot rolling beta coefficient 
plt.figure(figsize=(10,6))
plt.plot(rolling_out.index, 
         rolling_out["r_NQ"], 
         label = "Rolling 180 min coefficient")
# add a reference line for OLS beta
plt.axhline(y = beta_ols, 
            color = 'red', 
            linestyle = '--', 
            label = "Full sample coefficient")
plt.title("Rolling OLS beta coefficient of SP returns on NQ returns")
plt.xlabel("Date")
plt.legend()
plt.grid() #Several strange spikes

# 3a) Rolling Granger causality


# lets write the function that would extract the p-values
# from the Granger causality test

def granger_pvalue(df: pd.DataFrame, # dataframe with data
                   col1: str, # caused variable
                   col2: str, # causing variable
                   maxlag: int) -> float: # maximum lag to test
    granger_test = grangercausalitytests(df[[col2, col1]], maxlag = [maxlag],
                                         verbose = False) # do NOT print the results
    p_value = granger_test[maxlag][0]['ssr_ftest'][1]
    return float(p_value)

# and apply it to our full dataset
p_value = granger_pvalue(data1_roll, 
                         "NQ", "SP", 
                         maxlag = 10)
print("p-value from Granger causality test (NQ causes SP):", p_value)

df_N_S = data1_roll[["NQ", "SP"]] #only quotation values
df_N_S

# ### STRATEGY SETUP WITH DIFFERENT ENTRY/EXIT STRATEGIES


data1 = pd.read_pickle("data1_input.pkl")
data1.head(401)

data1.between_time("15:51", "16:00").head(21)

# Let's see the effect on the plot

# Let's convert the datetime index to text and pass it as the x-axis
data1_plot = data1.copy()

data1_plot['time'] = data1_plot.index.astype(str)

# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop = True)

# Set time as the X-axis and plot the charts
data1_plot.plot(
    x = 'time',
    subplots = True,
    layout = (2, 1),
    title = "Quotations of NQ and SP"
)

plt.show()


# a) SIMPLE MOVING AVERAGE (SMA)


# let's calculate the 20-minute moving average for NQ price

data1['NQ_SMA20'] = data1['NQ'].rolling(window = 20, 
                                                                                # lets assume we need at least 50%
                                                                                # of non-missing values to calculate the mean
                                                                                min_periods = 10).mean()

# let's insert missing values in the moving average
# whenever the original price is missing
# (this applies to the first and last 5 minutes of each session)

data1['NQ_SMA20'] = data1['NQ_SMA20'].where(
    # keep the value only if the original price is not missing
    # otherwise set it to NaN
    ~data1['NQ'].isna(), np.nan)


# Let's compare the SMA on the chart with the original closing price
# for the first ten days of data

# select only the columns needed for the plot
dat1_plot = data1[['NQ',
                                            'NQ_SMA20']].copy()
# Filter only the first 6 days
end_date = dat1_plot.index.min() + pd.Timedelta(days = 5)
dat1_plot = dat1_plot.loc[:end_date]

# Add a time column (index as datetime)
dat1_plot['time'] = dat1_plot.index

# Reset the index so that 'time' is a column
dat1_plot = dat1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))
plt.plot(dat1_plot.index, 
         dat1_plot['NQ'], 
         label = 'Price', 
         color='gray')
plt.plot(dat1_plot.index, 
         dat1_plot['NQ_SMA20'], 
         label = 'SMA20', 
         color = 'blue',  
         linewidth = 2)

# mark on the X axis ticks with dates/times every half hour
tick_mask = dat1_plot['time'].dt.minute % 30 == 0
xticks = dat1_plot.index[tick_mask]
xticklabels = dat1_plot['time'][tick_mask].dt.strftime('%Y-%m-%d %H:%M')
plt.xticks(ticks = xticks, labels = xticklabels, rotation = 90, ha = 'right')

plt.title("Quotations of NQ and 20-min SMA")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show() #pretty good

# b) 2 SMA


# let's calculate the 10-minute moving average for NQ price
data1['NQ_SMA10'] = data1['NQ'].rolling(window = 10, 
                                                                                min_periods=5).mean()

# let's insert missing values in the moving average
# whenever the original price is missing
# (this applies to the first and last 5 minutes of each session)

data1['NQ_SMA10'] = data1['NQ_SMA10'].where(
    ~data1['NQ'].isna(), np.nan)

# let's calculate the 30-minute moving average for NQ price
data1['NQ_SMA30'] = data1['NQ'].rolling(window = 30, 
                                                                                min_periods = 15).mean()
# and insert missing values in the moving average
data1['NQ_SMA30'] = data1['NQ_SMA30'].where(
    ~data1['NQ'].isna(), np.nan)

# Let's compare both SMAs on the chart with the original closing price

# Let's select only the columns needed for the chart
data1_plot = data1[['NQ',
                                            'NQ_SMA10',
                                            'NQ_SMA30']].copy()
# Filter only the first 6 days
end_date = data1_plot.index.min() + pd.Timedelta(days = 5)
data1_plot = data1_plot.loc[:end_date]

# Add a time column (index as datetime)
data1_plot['time'] = data1_plot.index

# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))
plt.plot(data1_plot.index, 
         data1_plot['NQ'], 
         label = 'Price', 
         color='gray')
plt.plot(data1_plot.index, 
         data1_plot['NQ_SMA10'], 
         label = 'SMA10', 
         color = 'blue',  
         linewidth = 2)
plt.plot(data1_plot.index, 
         data1_plot['NQ_SMA30'], 
         label = 'SMA30', 
         color = 'red',  
         linewidth = 2)

# mark on the X axis ticks with dates/times every half hour
tick_mask = data1_plot['time'].dt.minute % 30 == 0
xticks = data1_plot.index[tick_mask]
xticklabels = data1_plot['time'][tick_mask].dt.strftime('%Y-%m-%d %H:%M')
plt.xticks(ticks = xticks, labels = xticklabels, rotation = 90, ha = 'right')

plt.title("Quotations of NQ and 2 SMAs (SMA10 and SMA30)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# c) 2 EXPONENTIAL MOVING AVERAGES (EMA)


# let's calculate the 10-minute EMA for the NQ price

# The easiest way to do this in Python is to use the .ewm() method 
# (Exponential Moving Window) from Pandas
# .ewm() parameters:
# - span: number of periods (the smaller, the faster the EMA)
# - alternative: halflife: half-life (the smaller, the faster the EMA)
# - min_periods: how much data is needed to start calculating (default = 0)

data1['NQ_EMA10'] = data1['NQ'].ewm(span = 10, 
                                                                            min_periods = 5).mean()

# let's insert missing values in the moving average
# each time the original price is missing
# (this applies to the first and last 5 minutes of each session)

data1['NQ_EMA10'] = data1['NQ_EMA10'].where(
    ~data1['NQ'].isna(), np.nan)

# and let's calculate the 30-minute EMA for the NQ price

data1['NQ_EMA30'] = data1['NQ'].ewm(span = 30, 
                                                                            min_periods = 15).mean()

# and insert missing values in the moving average
# each time the original price is missing
data1['NQ_EMA30'] = data1['NQ_EMA30'].where(
    ~data1['NQ'].isna(), np.nan)


# Let's see them on the price chart

# Let's select only the columns needed for the chart
data1_plot = data1[['NQ',
                                            'NQ_EMA10',
                                            'NQ_EMA30']].copy()
# Filter only the first 6 days
end_date = data1_plot.index.min() + pd.Timedelta(days = 5)
dataUSA_AAPL_META_plot = data1_plot.loc[:end_date]

# Add a time column (index as datetime)
data1_plot['time'] = data1_plot.index

# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))
plt.plot(data1_plot.index, 
         data1_plot['NQ'], 
         label = 'Price', 
         color='gray')
plt.plot(data1_plot.index, 
         data1_plot['NQ_EMA10'], 
         label = 'EMA10', 
         color = 'green',  
         linewidth = 2)
plt.plot(data1_plot.index, 
         data1_plot['NQ_EMA30'], 
         label = 'EMA30', 
         color = 'orange',  
         linewidth = 2)

# mark on the X axis ticks with dates/times every half hour
tick_mask = data1_plot['time'].dt.minute % 30 == 0
xticks = data1_plot.index[tick_mask]
xticklabels = data1_plot['time'][tick_mask].dt.strftime('%Y-%m-%d %H:%M')
plt.xticks(ticks = xticks, labels = xticklabels, rotation = 90, ha = 'right')

plt.title("Quotations of NQ and 2 EMAs (EMA10 i EMA30)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Let's compare the EMA10 with the SMA10

# Let's select only the columns needed for the chart
data1_plot = data1[['NQ_SMA10',
                                            'NQ_EMA10']].copy()

# Filter only the first 3 days
end_date = data1_plot.index.min() + pd.Timedelta(days = 2)
data1_plot = data1_plot.loc[:end_date]
# Add a time column (index as datetime)
data1_plot['time'] = data1_plot.index
# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))
plt.plot(data1_plot.index, 
         data1_plot['NQ_SMA10'], 
         label = 'SMA10', 
         color = 'blue',  
         linewidth = 2)
plt.plot(data1_plot.index, 
         data1_plot['NQ_EMA10'],
            label = 'EMA10',
            color = 'green',  
            linewidth = 2)

# Let's add a legend
plt.legend()

# Let's compare the EMA30 with the SMA30 now

# Let's select only the columns needed for the chart
data1_plot = data1[['NQ_SMA30',
                                            'NQ_EMA30']].copy()

# Filter only the first 6 days
end_date = data1_plot.index.min() + pd.Timedelta(days = 5)
data1_plot = data1_plot.loc[:end_date]
# Add a time column (index as datetime)
data1_plot['time'] = data1_plot.index
# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))
plt.plot(data1_plot.index, 
         data1_plot['NQ_SMA30'], 
         label = 'SMA10', 
         color = 'red',  
         linewidth = 2)
plt.plot(data1_plot.index, 
         data1_plot['NQ_EMA30'],
            label = 'EMA10',
            color = 'orange',  
            linewidth = 2)

# Let's add a legend
plt.legend()

# with a longer memory, it is more visible that 
# the EMA reacts faster to price changes than the SMA

# d) VOLATILITY BREAKOUT MODELS


# let's calculate the 60-minute standard deviation of the nq price

data1['NQ_STD30'] = data1['NQ'].rolling(window = 30,
                                                                                min_periods = 15).std()

# Let's see the closing price and thresholds on the chart
# determined based on volatility
# SMA60+/-1.5*rollstd60

# Select only the columns needed for the chart
data1_plot = data1[['NQ',
                                            'NQ_SMA30',
                                            'NQ_STD30']].copy()

# Filter only the first 3 days
end_date = data1_plot.index.min() + pd.Timedelta(days = 2)
data1_plot = data1_plot.loc[:end_date]
# Add a time column (index as datetime)
data1_plot['time'] = data1_plot.index
# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))
plt.plot(data1_plot.index, 
         data1_plot['NQ'], 
         color = 'lime',  
         linewidth = 1)
plt.plot(data1_plot.index, 
         data1_plot['NQ_SMA30'] + 1.5 * data1_plot['NQ_STD30'], 
         color = 'darkgreen',  
         linewidth = 2)
plt.plot(data1_plot.index, 
         data1_plot['NQ_SMA30'] - 1.5 * data1_plot['NQ_STD30'],
         color = 'darkgreen',  
         linewidth = 2)


# let's calculate an alternative measure of variability - Mean Absolute Deviation (MAD)

data1['NQ_MAD30'] = data1['NQ'].rolling(window = 30,
                                                                                min_periods = 15).apply(
    lambda x: np.mean(np.abs(x - np.mean(x))), raw = True)

# and also Median Absolute Deviation (MedAD)

data1['NQ_MedAD30'] = data1['NQ'].rolling(window = 30,
                                                                                  min_periods = 15).apply(
    lambda x: np.median(np.abs(x - np.median(x))), raw = True)


# Let's compare three alternative measures of volatility on a chart

# Select only the columns needed for the chart
data1_plot = data1[['NQ_STD30',
                                            'NQ_MAD30',
                                            'NQ_MedAD30']].copy()

# Filter only the first 3 days
end_date = data1_plot.index.min() + pd.Timedelta(days = 2)
data1_plot = data1_plot.loc[:end_date]
# Add a time column (index as datetime)
data1_plot['time'] = data1_plot.index
# Reset the index so that 'time' is a column
data1_plot = data1_plot.reset_index(drop=True)

# We make a plot by observation number – so that there are no gaps between sessions
plt.figure(figsize = (12, 6))   
plt.plot(data1_plot.index, 
         data1_plot['NQ_STD30'], 
         label = 'STD30', 
         color = 'blue',  
         linewidth = 2)
plt.plot(data1_plot.index,
            data1_plot['NQ_MAD30'], 
            label = 'MAD30', 
            color = 'orange',  
            linewidth = 2)
plt.plot(data1_plot.index,
            data1_plot['NQ_MedAD30'], 
            label = 'MedAD30', 
            color = 'red',  
            linewidth = 2)

# Let's add a legend
plt.legend()

# similar volatility patterns, but alternative measures
# can be used as one of the strategy parameters

#Let's also add rolling window=60 for each approach and further comparison

data1['NQ_SMA60'] = data1['NQ'].rolling(window = 60, 
                                                                                min_periods = 30).mean()

data1['NQ_EMA60'] = data1['NQ'].ewm(span = 60, 
                                                                            min_periods = 30).mean()

data1['NQ_STD60'] = data1['NQ'].rolling(window = 60,
                                                                                min_periods = 30).std()

data1['NQ_MAD60'] = data1['NQ'].rolling(window = 60,
                                                                                min_periods = 30).apply(
    lambda x: np.mean(np.abs(x - np.mean(x))), raw = True)

# and also Median Absolute Deviation (MedAD)

data1['NQ_MedAD60'] = data1['NQ'].rolling(window = 60,
                                                                                  min_periods = 30).apply(
    lambda x: np.median(np.abs(x - np.median(x))), raw = True)

data1

#Saving the results
data1.to_parquet("data1_SMA_EMA.parquet")

# ### CALCULATING POSITION AND PNL OF DEVELOPPED STRATEGIES


#Loading data

data1 = pd.read_parquet('data1_SMA_EMA.parquet')
data1.head(401)

# Assumptions for NQ futures:
# NQ – futures contract for NASDAQ index (transaction cost = 12$, point value = 20$).

# 1) Single SMA20


# Let's calculate the position for the MOMENTUM strategy:
# - if price(t-1) > SMA20(t-1) => pos(t) = 1 [long]
# - if price(t-1) <= SMA20(t-1) => pos(t) = -1 [short]

# We can easily read the data from the 
# previous observation using .shift(1)

cond1_mom_long = data1['NQ'].shift(1) > data1['NQ_SMA20'].shift(1)

# We put 1 where the condition is met,
# -1 where it is not met

data1['position1_mom'] = np.where(cond1_mom_long, 1, -1)

# NOTE! This strategy always has a position!
# (always in the market)

# Let's calculate the position for the MEAN-REVERTING strategy:
# - if price(t-1) > SMA20(t-1) => pos(t) = -1 [short]
# - if price(t-1) <= SMA20(t-1) => pos(t) = 1 [long]

# For simplicity, we can use the same condition.

# Now we put -1 where the condition is met,
# 1 where it is not met.

data1['position1_mr'] = np.where(cond1_mom_long, -1, 1)

data1

# We need to additionally check whether we're comparing
# NaN values, because Python doesn't automatically check for this

# Let's add filters that check for NaN values
lagprice_nonmiss = data1['NQ'].shift(1).notna()
lagsma_nonmiss = data1['NQ_SMA20'].shift(1).notna()

# Now we can add these conditions to our strategies.
# If any of the values ​​are missing,
# then we cannot make a position decision

data1['position1_mom'] = np.where(
        lagprice_nonmiss & lagsma_nonmiss,
        np.where(cond1_mom_long, 1, -1),
        np.nan)

data1['position1_mr'] = np.where(
        lagprice_nonmiss & lagsma_nonmiss,
        np.where(cond1_mom_long, -1, 1),
        np.nan)

# let's check if everything works
data1.tail(10)

# now it is better

# Let's also assume that:
# - We ALWAYS exit positions 20 minutes before the end of the session (at 3:40 PM) -> signal=(-1) at 15:40
# - We NEVER enter positions in the first 25 minutes of the session (until 9:55 AM) -> signal=0

# We change the position to 0 for the 9:30–9:55 AM range

data1.loc[
    data1.between_time("09:30", "09:55").index,
    ['position1_mom', 'position1_mr']
] = 0

# For the interval 15:40–16:00
data1.loc[
    data1.between_time("15:41", "16:00").index,
    ['position1_mom', 'position1_mr']
] = 0

data1.tail(20)

# Let's calculate the gross PnL for both strategies

# We'll calculate the price change for NQ at a given moment
# using the diff() function, which returns the difference
# between the current and previous values

# MOMENTUM strategy:
data1['pnl_gross1_mom'] = (
    data1['NQ'].diff() * 
    data1['position1_mom'])

# similarly for the mean reversion strategy
data1['pnl_gross1_mr'] = (
    data1['NQ'].diff() * 
    data1['position1_mr'])

# we replace missing values with 0
data1['pnl_gross1_mom'] = data1['pnl_gross1_mom'].fillna(0)
data1['pnl_gross1_mr'] = data1['pnl_gross1_mr'].fillna(0)

# Let's draw a cumulative PnL chart without adding any columns to the data

# Let's create a copy of the data for the chart
data1_plot = data1.copy()

# Let's convert the index to a column so we can 
# draw a graph without gaps in the time data.
data1_plot.reset_index(inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross1_mom']), 
    color='blue')
plt.title("Cumulative gross PnL of the MOMENTUM strategy")
plt.grid()

# Let's see what the cumulative PnL looks like
# for the MEAN-REVERTING strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1.index, 
    np.cumsum(data1['pnl_gross1_mr']), 
    color='blue')
plt.title("Cumulative gross PnL of the MEAN-REVERTING strategy")
plt.grid()

# Let's calculate the number of transactions (this is simply the absolute value
# of position changes in a given minute) - it will be identical
# for the MOMENTUM and MEAN-REVERTING strategies
# based on the same data

data1['ntrans1'] = data1['position1_mom'].diff().abs()

# Given the number of transactions per minute,
# we can calculate the net PnL.

# we assume each transaction costs $12.

# for the MOMENTUM strategy
data1['pnl_net1_mom'] = (
    data1['pnl_gross1_mom'] -
    12 * data1['ntrans1'])

#  and for the MEAN-REVERTING approach
data1['pnl_net1_mr'] = (
    data1['pnl_gross1_mr'] -
    12* data1['ntrans1'])

# Let's plot the cumulative gross and net PnL for the MEAN-REVERTING 
# strategy (only this one was profitable in the gross sense)

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross1_mr']), 
    color='blue', label='Gross PnL')
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_net1_mr']), 
    color='red', label='Net PnL')
plt.title("Cumulative PnL of the MEAN-REVERTING strategy")
plt.legend()
plt.grid()

# no chance of profiting from this strategy...

# How about MOMENTUM strategy?

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross1_mom']), 
    color='blue', label='Gross PnL')
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_net1_mom']), 
    color='red', label='Net PnL')
plt.title("Cumulative PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# still big loss ...

# let's calculate the cumulative costs over the entire period

data1['ntrans1'].sum() * 12

# approximately 433 k $

# Let's load the position visualization function
# prepared by the lecturer

from functions.plot_positions import plot_positions_ma

# let's see the strategy activity for the selected day

plot_positions_ma(
    data_plot = data1, # DataFrame with DatetimeIndex
    date_plot = '2024-04-01',      # Date as string 'YYYY-MM-DD'
    col_price = 'NQ',      # column with price
    col_ma = 'NQ_SMA20',   # column with moving average/median
    col_pos = 'position1_mr',      # column with position (-1, 0, 1)
    title = 'MEAN-REVERTING strategy activity for AAPL 01/04/2024')

# the function also has a save_graph argument (default: False),
# which saves the graph to a .png file
# you must also provide the file name in the file_name argument

# the strategy changes positions quite often...

# let's see another day

plot_positions_ma(
    data_plot = data1, # DataFrame with DatetimeIndex
    date_plot = '2024-05-01',      # Date as string 'YYYY-MM-DD'
    col_price = 'NQ',      # column with price
    col_ma = 'NQ_SMA20',   # column with moving average/median
    col_pos = 'position1_mr',      # column with position (-1, 0, 1)
    title = 'MEAN-REVERTING strategy activity for AAPL 01/04/2024')

# the function also has a save_graph argument (default: False),
# which saves the graph to a .png file
# you must also provide the file name in the file_name argument

# the strategy changes positions quite often...

# 2b) 2 EMA (EMA60, EMA10)


# Let's calculate the position for the MOMENTUM strategy
# if fast MA(t-1) > slow MA(t-1) => pos(t) = 1 [long]
# if fast MA(t-1) <= slow MA(t-1) => pos(t) = -1 [short]

# NOTE: This strategy also always has a position! (no flat- for now only on restricted times)

cond2b_mom_long = data1['NQ_EMA10'].shift(1) > data1['NQ_EMA60'].shift(1)

# let's add filters that will check for NaN values
lagsema10_nonmiss = data1['NQ_EMA10'].shift(1).notna()
lagsema60_nonmiss = data1['NQ_EMA60'].shift(1).notna()

# Now we can add these conditions to our strategies
# If any of the values ​​are missing,
# then we cannot make a position decision

data1['position2b_mom'] = np.where(
        lagsema10_nonmiss & lagsema60_nonmiss,
        np.where(cond2b_mom_long, 1, -1),
        np.nan)

data1['position2b_mr'] = np.where(
        lagsema10_nonmiss & lagsema60_nonmiss,
        np.where(cond2b_mom_long, -1, 1),
        np.nan)
        

# Let's also assume that:
# - We ALWAYS exit positions 20 minutes before the end of the session (at 3:40 PM)
# - We NEVER enter positions in the first 25 minutes of the session (until 9:55 AM)

# We change the position to 0 for the 9:30–9:55 AM range

data1.loc[
    data1.between_time("09:30", "09:55").index,
    ['position2b_mom', 'position2b_mr']
] = 0

# For the interval 15:40–16:00
data1.loc[
    data1.between_time("15:41", "16:00").index,
    ['position2b_mom', 'position2b_mr']
] = 0

data1.tail(20)

# Calculating gross and net PnL for the strategy

# Gross
# MOMENTUM strategy:
data1['pnl_gross2b_mom'] = (
    data1['NQ'].diff() * 
    data1['position2b_mom'])

# similarly for the MEAN-REVERTING strategy
data1['pnl_gross2b_mr'] = (
    data1['NQ'].diff() * 
    data1['position2b_mr'])

# we replace missing values with 0
data1['pnl_gross2b_mom'] = data1['pnl_gross2b_mom'].fillna(0)
data1['pnl_gross2b_mr'] = data1['pnl_gross2b_mr'].fillna(0)

# number of transactions
data1['ntrans2b'] = data1['position2b_mom'].diff().abs()

# Net PnL
# for the MOMENTUM strategy
data1['pnl_net2b_mom'] = (
    data1['pnl_gross2b_mom'] -
    12 * data1['ntrans2b'])

# for the MEAN-REVERTING strategy
data1['pnl_net2b_mr'] = (
    data1['pnl_gross2b_mr'] -
    12 * data1['ntrans2b'])

# let's draw the cumulative gross PnL for the MOMENTUM strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross2b_mom']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# I don't think you can make any money here (neither MOM nor MR)...

# let's draw the cumulative gross PnL for the MEAN-REVERTING strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross2b_mr']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MEAN-REVERTING strategy")
plt.legend()
plt.grid()

# I don't think you can make any money here (neither MOM nor MR)...

# Let's add the cumulative net PnL for the MOMENTUM strategy
# (only this one was profitable in the gross sense)

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross2b_mom']), 
    color='blue', label='Gross PnL')
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_net2b_mom']), 
    color='red', label='Net PnL')
plt.title("Cumulative PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# even higher loss

# let's calculate the cumulative costs of this strategy over the entire period

data1['ntrans2b'].sum() * 12

# over $95 k - 4-5 times more than in the single MA strategy,
# but still a lot

from functions.plot_positions import plot_positions_2mas

plot_positions_2mas(
    data_plot=data1,
    date_plot='2024-04-01',
    col_price='NQ',
    col_fma='NQ_EMA10',
    col_sma='NQ_EMA60',
    col_pos='position2b_mom',
    title='MOMENTUM strategy 2MAs for NQ - 01/04/2024',
    save_graph=False
)

# This strategy changes positions less frequently than 
# the previous one, but still too often to be profitable.

#Another day
plot_positions_2mas(
    data_plot=data1,
    date_plot='2024-05-01',
    col_price='NQ',
    col_fma='NQ_EMA10',
    col_sma='NQ_EMA60',
    col_pos='position2b_mom',
    title='MOMENTUM strategy 2MAs for NQ - 01/05/2024',
    save_graph=False
)


# 2a) 2 EMA (EMA60, EMA30)


# Let's calculate the position for the MOMENTUM strategy
# if fast MA(t-1) > slow MA(t-1) => pos(t) = 1 [long]
# if fast MA(t-1) <= slow MA(t-1) => pos(t) = -1 [short]

# NOTE: This strategy also always has a position! (no flat- for now only on restricted times)

cond2a_mom_long = data1['NQ_EMA30'].shift(1) > data1['NQ_EMA60'].shift(1)

# let's add filters that will check for NaN values
lagsema30_nonmiss = data1['NQ_EMA30'].shift(1).notna()
lagsema60_nonmiss = data1['NQ_EMA60'].shift(1).notna()

# Now we can add these conditions to our strategies
# If any of the values ​​are missing,
# then we cannot make a position decision

data1['position2a_mom'] = np.where(
        lagsema30_nonmiss & lagsema60_nonmiss,
        np.where(cond2a_mom_long, 1, -1),
        np.nan)

data1['position2a_mr'] = np.where(
        lagsema30_nonmiss & lagsema60_nonmiss,
        np.where(cond2a_mom_long, -1, 1),
        np.nan)
        

# Let's also assume that:
# - We ALWAYS exit positions 20 minutes before the end of the session (at 3:40 PM)
# - We NEVER enter positions in the first 25 minutes of the session (until 9:55 AM)

# We change the position to 0 for the 9:30–9:55 AM range

data1.loc[
    data1.between_time("09:30", "09:55").index,
    ['position2a_mom', 'position2a_mr']
] = 0

# For the interval 15:40–16:00
data1.loc[
    data1.between_time("15:41", "16:00").index,
    ['position2a_mom', 'position2a_mr']
] = 0

data1.tail(20)

# Calculating gross and net PnL for the strategy

# Gross
# MOMENTUM strategy:
data1['pnl_gross2a_mom'] = (
    data1['NQ'].diff() * 
    data1['position2a_mom'])

# similarly for the MEAN-REVERTING strategy
data1['pnl_gross2a_mr'] = (
    data1['NQ'].diff() * 
    data1['position2a_mr'])

# we replace missing values with 0
data1['pnl_gross2a_mom'] = data1['pnl_gross2a_mom'].fillna(0)
data1['pnl_gross2a_mr'] = data1['pnl_gross2a_mr'].fillna(0)

# number of transactions
data1['ntrans2a'] = data1['position2a_mom'].diff().abs()

# Net PnL
# for the MOMENTUM strategy
data1['pnl_net2a_mom'] = (
    data1['pnl_gross2a_mom'] -
    12 * data1['ntrans2a'])

# for the MEAN-REVERTING strategy
data1['pnl_net2a_mr'] = (
    data1['pnl_gross2a_mr'] -
    12 * data1['ntrans2a'])

# let's draw the cumulative gross PnL for the MOMENTUM strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross2a_mom']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# Even lower gross PnL value

# let's draw the cumulative gross PnL for the MEAN-REVERTING strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross2a_mr']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MEAN-REVERTING strategy")
plt.legend()
plt.grid()

# Even lower gross PnL value

# Let's add the cumulative net PnL for the MOMENTUM strategy
# (only this one was profitable in the gross sense)

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross2a_mom']), 
    color='blue', label='Gross PnL')
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_net2a_mom']), 
    color='red', label='Net PnL')
plt.title("Cumulative PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# closer, closer to net=0 at least ...

# let's calculate the cumulative costs of this strategy over the entire period

data1['ntrans2a'].sum() * 12

# in between with single SMA/double EMAs

plot_positions_2mas(
    data_plot=data1,
    date_plot='2024-04-01',
    col_price='NQ',
    col_fma='NQ_EMA30',
    col_sma='NQ_EMA60',
    col_pos='position2a_mom',
    title='MOMENTUM strategy 2MAs for NQ - 01/04/2024',
    save_graph=False
)


plot_positions_2mas(
    data_plot=data1,
    date_plot='2024-05-01',
    col_price='NQ',
    col_fma='NQ_EMA30',
    col_sma='NQ_EMA60',
    col_pos='position2a_mom',
    title='MOMENTUM strategy 2MAs for NQ - 01/05/2024',
    save_graph=False
)
 #only 2 signals

# 3b) 2 SMA (SMA60, SMA10)


# Let's calculate the position for the MOMENTUM strategy
# if fast MA(t-1) > slow MA(t-1) => pos(t) = 1 [long]
# if fast MA(t-1) <= slow MA(t-1) => pos(t) = -1 [short]

# NOTE: This strategy also always has a position! (no flat- for now only on restricted times)

cond3b_mom_long = data1['NQ_SMA10'].shift(1) > data1['NQ_SMA60'].shift(1)

# let's add filters that will check for NaN values
lagssma10_nonmiss = data1['NQ_SMA10'].shift(1).notna()
lagssma60_nonmiss = data1['NQ_SMA60'].shift(1).notna()

# Now we can add these conditions to our strategies
# If any of the values ​​are missing,
# then we cannot make a position decision

data1['position3b_mom'] = np.where(
        lagssma10_nonmiss & lagssma60_nonmiss,
        np.where(cond3b_mom_long, 1, -1),
        np.nan)

data1['position3b_mr'] = np.where(
        lagssma10_nonmiss & lagssma60_nonmiss,
        np.where(cond3b_mom_long, -1, 1),
        np.nan)
        

# Let's also assume that:
# - We ALWAYS exit positions 20 minutes before the end of the session (at 3:40 PM)
# - We NEVER enter positions in the first 25 minutes of the session (until 9:55 AM)

# We change the position to 0 for the 9:30–9:55 AM range

data1.loc[
    data1.between_time("09:30", "09:55").index,
    ['position3b_mom', 'position3b_mr']
] = 0

# For the interval 15:40–16:00
data1.loc[
    data1.between_time("15:41", "16:00").index,
    ['position3b_mom', 'position3b_mr']
] = 0

data1.tail(20)

# Calculating gross and net PnL for the strategy

# Gross
# MOMENTUM strategy:
data1['pnl_gross3b_mom'] = (
    data1['NQ'].diff() * 
    data1['position3b_mom'])

# similarly for the MEAN-REVERTING strategy
data1['pnl_gross3b_mr'] = (
    data1['NQ'].diff() * 
    data1['position3b_mr'])

# we replace missing values with 0
data1['pnl_gross3b_mom'] = data1['pnl_gross3b_mom'].fillna(0)
data1['pnl_gross3b_mr'] = data1['pnl_gross3b_mr'].fillna(0)

# number of transactions
data1['ntrans3b'] = data1['position3b_mom'].diff().abs()

# Net PnL
# for the MOMENTUM strategy
data1['pnl_net3b_mom'] = (
    data1['pnl_gross3b_mom'] -
    12 * data1['ntrans3b'])

# for the MEAN-REVERTING strategy
data1['pnl_net3b_mr'] = (
    data1['pnl_gross3b_mr'] -
    12 * data1['ntrans3b'])

# let's draw the cumulative gross PnL for the MOMENTUM strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross3b_mom']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# nice one

# let's draw the cumulative gross PnL for the MEAN-REVERTING strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross3b_mr']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MEAN-REVERTING strategy")
plt.legend()
plt.grid()


# Let's add the cumulative net PnL for the MOMENTUM strategy
# (only this one was profitable in the gross sense)

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross3b_mom']), 
    color='blue', label='Gross PnL')
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_net3b_mom']), 
    color='red', label='Net PnL')
plt.title("Cumulative PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# but one can see: transactional costs 'eat' all the profit

# let's calculate the cumulative costs of this strategy over the entire period

data1['ntrans3b'].sum() * 12

# 4) VOLATILITY BREAKOUT


# Our strategy's signal will be the NQ price
# (it can also be the NQ fast moving average -
# faster than 60 minutes, which was used in the definition of
# entry/exit boundaries)

# Let's calculate the position for the MOMENTUM strategy
# (a more complicated formula than before)
# if pos(t-1) = 0 and
#   signal(t-1) > upper threshold(t-1) => pos(t) = 1 [long]
#   signal(t-1) < lower threshold(t-1) => pos(t) = -1 [short]
#   otherwise keep 0 position [flat]
# if pos(t-1) = 1 and
#   signal(t-1) > lower threshold(t-1) => pos(t) = 1 [keep long]
#   signal(t-1) < lower threshold(t-1) => pos(t) = -1 [switch to short]
# if pos(t-1) = -1 and
#   signal(t-1) < upper threshold(t-1) => pos(t) = -1 [keep short]
#   signal(t-1) > upper threshold(t-1) => pos(t) = 1 [switch to long]

# NOTE! This strategy always has a position if
# the entry and exit thresholds have the same multiplier!

# we will use a function written by the lecturer for this purpose

from functions.position_VB import positionVB

# function arguments:
# signal: strategy signal
# lower: lower threshold
# upper: upper threshold
# pos_flat - vector indicating when the position must be 0 (flat)
# strategy: strategy type: "mom" or "mr"

# Let's create a pos_flat vector that will indicate
# when the position must be 0 (flat)

# Let's first initialize it with zeros
pos_flat = np.zeros(len(data1))

# We assumed that we didn't want to take positions in the first 25 minutes
# and last 200 minutes of the session (until 9:40 and from 15:51)
pos_flat[(data1.index.time >= pd.to_datetime('15:41').time())] = 1
pos_flat[(data1.index.time <= pd.to_datetime('09:55').time())] = 1

# We only have weekdays in the data, 
# so we don't need to check weekends.

# let's see the frequencies of values ​​in this vector
pd.Series(pos_flat).value_counts()

# let's calculate the position for the MOMENTUM strategy
data1['position4_mom'] = positionVB(
    signal = data1['NQ'],  # NQ price is the signal
    lower = data1['NQ_SMA60'] - data1['NQ_STD60'] * 3,  # lower threshold
    upper = data1['NQ_SMA60'] + data1['NQ_STD60'] * 3,  # upper threshold
    pos_flat = pos_flat,  # flat position vector
    strategy = 'mom'  # MOMENTUM strategy
)

# let's calculate the position for the MEAN-REVERTING strategy
# for the same data it will simply be
# the opposite of the MOMENTUM position
data1['position4_mr'] = - data1['position4_mom']


# Calculating gross and net PnL for strategy 4

# Gross
# MOMENTUM strategy:
data1['pnl_gross4_mom'] = (
    data1['NQ'].diff() * 
    data1['position4_mom'])

# similarly for the MEAN-REVERTING strategy
data1['pnl_gross4_mr'] = (
    data1['NQ'].diff() * 
    data1['position4_mr'])

# we replace missing values with 0
data1['pnl_gross4_mom'] = data1['pnl_gross4_mom'].fillna(0)
data1['pnl_gross4_mr'] = data1['pnl_gross4_mr'].fillna(0)

# number of transactions
data1['ntrans4'] = data1['position4_mom'].diff().abs()

# Net PnL
# for the MOMENTUM strategy
data1['pnl_net4_mom'] = (
    data1['pnl_gross4_mom'] -
    1.5 * data1['ntrans4'])

# for the MEAN-REVERTING strategy
data1['pnl_net4_mr'] = (
    data1['pnl_gross4_mr'] -
    1.5 * data1['ntrans4'])

# Let's plot the cumulative gross PnL for the MOMENTUM strategy

plt.figure(figsize=(12, 6))
plt.plot(
    data1_plot.index, 
    np.cumsum(data1['pnl_gross4_mom']), 
    color='blue', label='Gross PnL')
plt.title("Cumulative gross PnL of the MOMENTUM strategy")
plt.legend()
plt.grid()

# seems to be profitable, how about net?

# let's calculate the cumulative costs of 
# this strategy over the entire period

data1['ntrans4'].sum() * 12

# lowest transactional costs than other models - 12936$

# let's load the function to visualize the position
from functions.plot_positions import plot_positions_vb

# the function requires referencing column names from the data frame

# let's create these for col_lower and col_upper
data1['NQ_VB_lower'] = (
    data1['NQ_SMA60'] - 
    data1['NQ_STD60'] * 3)

data1['NQ_VB_upper'] = (
    data1['NQ_SMA60'] +
    data1['NQ_STD60'] * 3)

# let's see the strategy activity for the selected day
plot_positions_vb(
    data_plot = data1, # DataFrame with DatetimeIndex
    date_plot = '2024-04-01',      # Date as string 'YYYY-MM-DD'
    col_signal = 'NQ',      # column with price
    col_lower = 'NQ_VB_lower',  # lower threshold
    col_upper = 'NQ_VB_upper',  # upper threshold
    col_pos = 'position4_mom',     # column with position (-1, 0, 1)
    title = 'VB MOMENTUM strategy activity for NQ 01/04/2024')

# only 1 signal?

# let's load the function to visualize the position
from functions.plot_positions import plot_positions_vb

# the function requires referencing column names from the data frame

# let's create these for col_lower and col_upper
data1['NQ_VB_lower'] = (
    data1['NQ_SMA60'] - 
    data1['NQ_STD60'] * 3)

data1['NQ_VB_upper'] = (
    data1['NQ_SMA60'] +
    data1['NQ_STD60'] * 3)

# let's see the strategy activity for the selected day
plot_positions_vb(
    data_plot = data1, # DataFrame with DatetimeIndex
    date_plot = '2024-05-01',      # Date as string 'YYYY-MM-DD'
    col_signal = 'NQ',      # column with price
    col_lower = 'NQ_VB_lower',  # lower threshold
    col_upper = 'NQ_VB_upper',  # upper threshold
    col_pos = 'position4_mom',     # column with position (-1, 0, 1)
    title = 'VB MOMENTUM strategy activity for NQ 01/05/2024')

# 2 signals

# let's save the data in its current form to the parquet file
data1.to_parquet('data1_entrymodels.parquet')

# ## FURTHER ASSESMENT OF STRATEGY PERFORMANCE
# 
# From previous section, volatility breakout model (as a SMA60 and STD60 transformation) seems to be the 'local champion'


data1 = pd.read_parquet("data1_entrymodels.parquet")
data1.head(401)

#Let's analyse previously designed breakout model

# Let's check the number of minutes in an average trading day -
# So the strategy works between 9:55 and 15:40.

# Let's save the number of minutes to the object.

TradingMinsInDay = (6.5 * 60 +15)

TradingMinsInDay

# Let's calculate the annualized Sharpe Ratio (SR)

# We will write a simple mySR function
# with 2 parameters:

# x: profit/loss series
# scale: scale, e.g., 252 for daily data

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

# Gross SR for MOMENTUM strategy based on breakout

mySR(x = data1['pnl_gross4_mom'],
     scale = 252 * TradingMinsInDay)

# positive

# net SR for the same strategy

mySR(data1['pnl_net4_mom'],
     scale = 252 * TradingMinsInDay)

# yaay, actually seems profitable

# Let's add the profit/loss columns in percentage terms
# (as a % of invested capital)

# Since we're assuming the purchase and sale of 1 NQ share (no fractions), 
# the invested capital is the NQ share price at the time 
# beginning of the interval.

# strategy 4
data1['pnl_gross4_mom_pct'] = (
    data1['pnl_gross4_mom'] /
    data1['NQ'].shift(1))
data1['pnl_net4_mom_pct'] = (
    data1['pnl_net4_mom'] /
    data1['NQ'].shift(1))
data1['pnl_gross4_mr_pct'] = (
    data1['pnl_gross4_mr'] /
    data1['NQ'].shift(1))
data1['pnl_net4_mr_pct'] = (
    data1['pnl_net4_mr'] /
    data1['NQ'].shift(1))

# Let's see if the SR calculated based on percentage profits/losses
# is the same as the SR calculated based on USD profits/losses

mySR(data1['pnl_gross4_mom_pct'],
     scale = 252 * TradingMinsInDay)

# net SR
mySR(data1['pnl_net4_mom_pct'],
        scale = 252 * TradingMinsInDay)

# values are very similar (even if some discrepancies at 2th decimal point, more for gross SR)

# for calculating various profitability measures
# let's aggregate the calculated profit/loss by days
# for all columns whose names start with "pnl_"
# or "ntrans_"

data1_daily = (
    data1.resample('D') # split by days
    # and aggregate for selected columns
    .agg({col: 'sum' for col in data1.columns if col.startswith('pnl_') or col.startswith('ntrans')})
)

data1_daily.head()

# Let's see the summary of strategy 1 in a MOMENTUM variant
# - equity curve, drawdown, and daily profits/losses

# !! IMPORTANT !!
# functions from qs library require a series of returns
# i.e., profit/loss in percentage terms w/o missing values

returns_strategy = data1_daily['pnl_gross4_mom_pct'].dropna()

qs.plots.snapshot(returns_strategy, 
                  title = "Snapshot of strategy 4 on NQ (gross)")

# function for SR from the quantstats package

sharpe = qs.stats.sharpe(returns_strategy)
# values will be slightly different from calculations on minute data

# alternative measures
calmar = qs.stats.calmar(returns_strategy)
sortino = qs.stats.sortino(returns_strategy)
omega = qs.stats.omega(returns_strategy)

print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Calmar Ratio: {calmar:.4f}")
print(f"Sortino Ratio: {sortino:.4f}")
print(f"Omega Ratio: {omega:.4f}") #i mean, pretty good eh?


# Calculation of the maximum drawdown for the entire period.

qs.stats.max_drawdown(returns_strategy)

# -4.8%

# Visualization of the drawdown ("underwater" plot)
qs.plots.drawdown(returns_strategy, figsize=(10, 6))

# Let's calculate the average daily number of transactions
# for each strategy (excluding weekends,
# because the stock market is closed then)

# Let's ignore weekends (Saturday = 5, Sunday = 6)
weekdays_only = data1_daily[data1_daily.index.dayofweek < 5]

avg_ntrans = weekdays_only[[col for col in weekdays_only.columns if col.startswith('ntrans')]].mean()
print("Average daily number of transactions for each strategy:")
print(avg_ntrans) #see the difference? Przydałyby sie wczesniej tez ploty dla porównania

# ### OPTIMIZATION OF STRATEGY


# #### NASDAQ FUTURES (NQ)


data1 = pd.read_pickle("data1_input.pkl")
data1.head(401)

data1.tail(11)

# Let's split the data into training and test sets
# let's assume that data for the first two years (2023, 2024) is for training
# and data for 2025 is for testing

# train: all data up to end of 2024
data1_train = data1[data1.index < "2025-01-01"]

# test: all data from 2025 onwards
data1_test = data1[data1.index >= "2025-01-01"]

data1_train.tail()

# Lets take NQ values

NQ = data1_train['NQ']

# check sample break in session

plt.figure(figsize = (10,6))
plt.plot(NQ[960:12000])

# Let's assume that we CLOSE ALL POSITIONS before the break (at 15:40).
# so finally, the position will always be 0 between 14:40 and 16:00.
# similarly as earlier, let's create an object named "pos_flat" 
# = 1 if position has to be flat (= 0) - we do not trade
# = 0 otherwise

# let's fill it first with zeros
pos_flat = np.zeros(len(NQ))

# put our assumptions into pos_flat
pos_flat = np.zeros(len(NQ))

t = NQ.index.time

breaks = (
    ((t >= pd.to_datetime("09:30").time()) & (t <= pd.to_datetime("09:55").time())) |
    ((t >= pd.to_datetime("15:40").time()) & (t <= pd.to_datetime("16:00").time()))
)

pos_flat[breaks] = 1


# let's see an example break in the session from the perspective of pos_flat

plt.figure(figsize = (10,6))
plt.plot(pos_flat[960:1600])

# lets check which weekdays we have data for

dweek_ = NQ.index.dayofweek + 1  # Adjust so that 0 = Monday, ..., 6 = Sunday
print(dweek_.value_counts())

# no Saturdays and Sundays in the data (perfect distribution?)

# lets create a time_ object (vector of times)

time_ = NQ.index.time

# and let's fill the pos_flat vector with ones for weekends

pos_flat[((dweek_ == 5) & (time_ > pd.to_datetime('16:00').time())) |      # end of Friday
          (dweek_ == 6) |                                                  # whole Saturday (just in case)
          ((dweek_ == 7) & (time_ <= pd.to_datetime('9:55').time()))] = 1 # beginning of Sunday

# Strategy 1: 2 intersecting moving averages


# we check various parameter combinations in a loop

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

fastEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 75, 90, 120, 150, 180, 240, 300, 360, 420]

# create a dataframe to store results
summary_all_2MAs = pd.DataFrame()

# Loop over different parameter combinations
for fastEMA in fastEMA_parameters:
    for slowEMA in slowEMA_parameters:
                
                # ensure that fastEMA is less than slowEMA
                if fastEMA >= slowEMA:
                    continue

                print(f"fastEMA = {fastEMA}, slowEMA = {slowEMA}")

                # We calculate the appropriate EMA
                fastEMA_values = NQ.ewm(span = fastEMA).mean()
                slowEMA_values = NQ.ewm(span = slowEMA).mean()

                # Insert NaNs wherever the original price is missing
                fastEMA_values[NQ.isna()] = np.nan
                slowEMA_values[NQ.isna()] = np.nan 

                # Calculate position for momentum strategy
                cond2b_mom_long = fastEMA_values.shift(1) > slowEMA_values.shift(1)
                
                # let's add filters that check for the presence of NaN values
                fastEMA_nonmiss = fastEMA_values.shift(1).notna()
                slowEMA_nonmiss = slowEMA_values.shift(1).notna()

                # Now we can add these conditions to our strategies
                # if any of the values is missing,
                # we cannot make a position decision

                pos_mom = np.where(fastEMA_nonmiss & slowEMA_nonmiss,
                                   np.where(cond2b_mom_long, 1, -1),
                                   np.nan)
                pos_mr = -pos_mom 

                # Set position to 0 where pos_flat is 1
                pos_mom[pos_flat == 1] = 0
                pos_mr[pos_flat == 1] = 0
                
                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * NQ.diff()), 0, pos_mom * NQ.diff() * 20) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * NQ.diff()), 0, pos_mr * NQ.diff() * 20) 
                # point value for NQ

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on NQ
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on NQ
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = NQ.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(NQ.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = NQ.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(NQ.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = NQ.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(NQ.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = NQ.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(NQ.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = NQ.index.time
                ntrans_d = ntrans.groupby(NQ.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect necessary results into one object
                summary = pd.DataFrame({
                    'fastEMA': fastEMA,
                    'slowEMA': slowEMA,
                    'period': '2023-2024',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append results to the summary
                summary_all_2MAs = pd.concat([summary_all_2MAs, summary], ignore_index=True)

# lets see top 5 stategies with respect to net_SR_mom
summary_all_2MAs.sort_values(by = 'net_PnL_mom',
                            ascending = False).head(5) #High profits and net SR ratios on momentum strategies (up to +31K $!)

# lets see top 5 stategies with respect to net_SR_mr
summary_all_2MAs.sort_values(by = 'net_PnL_mr',
                            ascending = False).head(5)

# lets import a function written by the lecturer
# to visualize the strategy results in a form of a heatmap
from functions.plot_heatmap import plot_heatmap

# we wil use it to visualize net SR values
# for the momentum strategy (MOM)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Momentum strategies)'
)

#Several combinations with positive net SR (90:150 the best)

# we wil use it to visualize net SR values
# for the mean-reverting strategy (MOM)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Mean-reverting strategies)'
)

# No positive SR


# How about net PnL for Momentum strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Momentum Strategy)'
) #Strategy with highest net SR leads to higher profit, results seems pretty reasonable

# How about net PnL for Mean-reverting strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Mean-reverting Strategy)'
) 

# Strategy 2: Volatility breakout


#Update of sharpe ratio so it won't divide by 0
def mySR(x, scale=252):
    mu = np.nanmean(x)
    sigma = np.nanstd(x)

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return np.sqrt(scale) * mu / sigma

# let's use fastEMA as the strategy signal
# and use slowEMA+/-m*std as volatility boundaries

from functions.position_VB import positionVB

# we check various parameter combinations in a loop

signalEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 90, 120, 150, 180, 200, 240, 300]
volat_sd_parameters = [60, 90, 120]
m_parameters = [1, 2, 3]

# create a dataframe to store results
summary_all_breakout = pd.DataFrame()

# loop over different parameter combinations
for signalEMA in signalEMA_parameters:
    print(f"signalEMA = {signalEMA}")
    for slowEMA in slowEMA_parameters:
        for volat_sd in volat_sd_parameters:
            for m in m_parameters:
               
                # We calculate the appropriate EMA
                signalEMA_values = NQ.ewm(span = signalEMA).mean().to_numpy()
                slowEMA_values = NQ.ewm(span = slowEMA).mean().to_numpy()
                                
                # We calculate the standard deviation
                volat_sd_values = NQ.rolling(window = volat_sd).std().to_numpy()

                # Insert NaNs wherever the original price is missing
                signalEMA_values[NQ.isna()] = np.nan
                slowEMA_values[NQ.isna()] = np.nan 
                volat_sd_values[NQ.isna()] = np.nan 

                # Calculate position for momentum strategy
                pos_mom = positionVB(signal = signalEMA_values, 
                                     lower = slowEMA_values - m * volat_sd_values,
                                     upper = slowEMA_values + m * volat_sd_values,
                                     pos_flat = pos_flat,
                                     strategy = "mom")
                
                pos_mr = -pos_mom 

                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * NQ.diff()), 0, pos_mom * NQ.diff() * 20) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * NQ.diff()), 0, pos_mr * NQ.diff() * 20) 
                # point value for NQ = 20$

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on NQ
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on NQ
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = NQ.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(NQ.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = NQ.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(NQ.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = NQ.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(NQ.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = NQ.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(NQ.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = NQ.index.time
                ntrans_d = ntrans.groupby(NQ.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect the necessary results into one object
                summary = pd.DataFrame({
                    'signalEMA': signalEMA,
                    'slowEMA': slowEMA,
                    'volat_sd': volat_sd,
                    'm': m,
                    'period': '2023-2024',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append the results to the summary
                summary_all_breakout = pd.concat([summary_all_breakout, summary], ignore_index=True)

    # it takes a while
    # approximately 11 (!) minutes

# check 10 strategies with the best net_SR_mom (momentum strategies)

summary_all_breakout.sort_values(by = 'net_PnL_mom', 
                                 ascending = False).head(10) # 31K profit, slightly lower than for 2xEMA

# check 10 strategies with the best net_SR_mr

summary_all_breakout.sort_values(by = 'net_PnL_mr', 
                                 ascending = False).head(10) #Here also not bat, 19K net profit

# summarize the results for the mom strategy
# in the form of a heatmap

# here we have four parameters
# signalEMA, slowEMA, volat_sd and m,
# so to present the results in the form of a heatmap,
# we need to combine them in pairs

summary_all_breakout["signalEMA_slowEMA"] = (
    summary_all_breakout["signalEMA"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["slowEMA"].astype(int).astype(str).str.zfill(3)
)

summary_all_breakout["volat_sd_m"] = (
    summary_all_breakout["volat_sd"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["m"].astype(str)
)

summary_all_breakout.head()

# now we can plot the heatmap
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mom',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# for mean-reverting
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mr',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# Momentum: the highest net SR are for signalEMA = 10 and slowEMA = 60

# let's check the sensitivity analysis of SR to m and volat_sd

# we select only rows with values of signalEMA and slowEMA
summary_all_breakout_wybrane = summary_all_breakout[
    (summary_all_breakout['signalEMA'] == 10) & (summary_all_breakout['slowEMA'] == 60)
]

# and we create a heatmap for them
plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'net_SR_mom',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# Rather not an anomally

# heatmap plot can also be used to analyze transaction frequency

plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'av_daily_ntrans',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Average Daily Number of Transactions (Momentum strategy)',
    cmap = "Blues")

    # in line with expectations
    # the larger the m, the fewer the transactions

# Testing of developped strategies


# Lets take NQ values

NQ_test = data1_test['NQ']

# check sample break in session

plt.figure(figsize = (10,6))
plt.plot(NQ[960:12000])

# put our assumptions into pos_flat
pos_flat = np.zeros(len(NQ_test))

t = NQ_test.index.time

breaks = (
    ((t >= pd.to_datetime("09:30").time()) & (t <= pd.to_datetime("09:55").time())) |
    ((t >= pd.to_datetime("15:40").time()) & (t <= pd.to_datetime("16:00").time()))
)

pos_flat[breaks] = 1


# let's see an example break in the session from the perspective of pos_flat

plt.figure(figsize = (10,6))
plt.plot(pos_flat[960:1600])

# lets check which weekdays we have data for

dweek_ = NQ_test.index.dayofweek + 1  # Adjust so that 0 = Monday, ..., 6 = Sunday
print(dweek_.value_counts())

# no Saturdays and Sundays in the data

# lets create a time_ object (vector of times)

time_ = NQ_test.index.time

# and let's fill the pos_flat vector with ones for weekends

pos_flat[((dweek_ == 5) & (time_ > pd.to_datetime('16:00').time())) |      # end of Friday
          (dweek_ == 6) |                                                  # whole Saturday (just in case)
          ((dweek_ == 7) & (time_ <= pd.to_datetime('9:55').time()))] = 1 # beginning of Sunday

# Strategy 1: 2 intersecting moving averages


# we check various parameter combinations in a loop

fastEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 75, 90, 120, 150, 180, 240, 300, 360, 420]

# create a dataframe to store results
summary_all_2MAs = pd.DataFrame()

# Loop over different parameter combinations
for fastEMA in fastEMA_parameters:
    for slowEMA in slowEMA_parameters:
                
                # ensure that fastEMA is less than slowEMA
                if fastEMA >= slowEMA:
                    continue

                print(f"fastEMA = {fastEMA}, slowEMA = {slowEMA}")

                # We calculate the appropriate EMA
                fastEMA_values = NQ_test.ewm(span = fastEMA).mean()
                slowEMA_values = NQ_test.ewm(span = slowEMA).mean()

                # Insert NaNs wherever the original price is missing
                fastEMA_values[NQ_test.isna()] = np.nan
                slowEMA_values[NQ_test.isna()] = np.nan 

                # Calculate position for momentum strategy
                cond2b_mom_long = fastEMA_values.shift(1) > slowEMA_values.shift(1)
                
                # let's add filters that check for the presence of NaN values
                fastEMA_nonmiss = fastEMA_values.shift(1).notna()
                slowEMA_nonmiss = slowEMA_values.shift(1).notna()

                # Now we can add these conditions to our strategies
                # if any of the values is missing,
                # we cannot make a position decision

                pos_mom = np.where(fastEMA_nonmiss & slowEMA_nonmiss,
                                   np.where(cond2b_mom_long, 1, -1),
                                   np.nan)
                pos_mr = -pos_mom 

                # Set position to 0 where pos_flat is 1
                pos_mom[pos_flat == 1] = 0
                pos_mr[pos_flat == 1] = 0
                
                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * NQ_test.diff()), 0, pos_mom * NQ_test.diff() * 20) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * NQ_test.diff()), 0, pos_mr * NQ_test.diff() * 20) 
                # point value for NQ

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on NQ
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on NQ
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = NQ_test.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(NQ_test.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = NQ_test.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(NQ_test.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = NQ_test.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(NQ_test.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = NQ_test.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(NQ_test.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = NQ_test.index.time
                ntrans_d = ntrans.groupby(NQ_test.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect necessary results into one object
                summary = pd.DataFrame({
                    'fastEMA': fastEMA,
                    'slowEMA': slowEMA,
                    'period': '2025',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append results to the summary
                summary_all_2MAs = pd.concat([summary_all_2MAs, summary], ignore_index=True)

# lets see top 5 stategies with respect to net_SR_mom
summary_all_2MAs.sort_values(by = 'net_PnL_mom',
                            ascending = False).head(5) #net profit with negative value

# lets see top 5 stategies with respect to net_SR_mr
summary_all_2MAs.sort_values(by = 'net_PnL_mr',
                            ascending = False).head(5) #actual jump of profit up to 71K ???

# we wil use it to visualize net SR values
# for the momentum strategy (MOM)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Momentum strategies)'
)

#Some different results

# we wil use it to visualize net SR values
# for the mean-reverting strategy (MR)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Mean-reverting strategies)'
)

#Mean-reverting on testing is better? inconsistent results with training though

# How about net PnL for MOM strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Momentum Strategy)'
) #Different results, bottom-right corner pretty consistent (slowEMA=420 ^ fastEMA=60 as safe pick)

# How about net PnL for MR strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Mean-reverting Strategy)'
) 

# Strategy 2: Volatility breakout


# lets use fastEMA as the strategy signal
# and use slowEMA+/-m*std as volatility boundaries

#Update of sharpe ratio so it won't divide by 0
def mySR(x, scale=252):
    mu = np.nanmean(x)
    sigma = np.nanstd(x)

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return np.sqrt(scale) * mu / sigma

from functions.position_VB import positionVB

# we check various parameter combinations in a loop

signalEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 90, 120, 150, 180, 200, 240, 300]
volat_sd_parameters = [60, 90, 120]
m_parameters = [1, 2, 3]

# create a dataframe to store results
summary_all_breakout = pd.DataFrame()

# loop over different parameter combinations
for signalEMA in signalEMA_parameters:
    print(f"signalEMA = {signalEMA}")
    for slowEMA in slowEMA_parameters:
        for volat_sd in volat_sd_parameters:
            for m in m_parameters:
               
                # We calculate the appropriate EMA
                signalEMA_values = NQ_test.ewm(span = signalEMA).mean().to_numpy()
                slowEMA_values = NQ_test.ewm(span = slowEMA).mean().to_numpy()
                                
                # We calculate the standard deviation
                volat_sd_values = NQ_test.rolling(window = volat_sd).std().to_numpy()

                # Insert NaNs wherever the original price is missing
                signalEMA_values[NQ_test.isna()] = np.nan
                slowEMA_values[NQ_test.isna()] = np.nan 
                volat_sd_values[NQ_test.isna()] = np.nan 

                # Calculate position for momentum strategy
                pos_mom = positionVB(signal = signalEMA_values, 
                                     lower = slowEMA_values - m * volat_sd_values,
                                     upper = slowEMA_values + m * volat_sd_values,
                                     pos_flat = pos_flat,
                                     strategy = "mom")
                
                pos_mr = -pos_mom 

                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * NQ_test.diff()), 0, pos_mom * NQ_test.diff() * 20) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * NQ_test.diff()), 0, pos_mr * NQ_test.diff() * 20) 
                # point value for NQ = 20$

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on NQ
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on NQ
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = NQ_test.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(NQ_test.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = NQ_test.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(NQ_test.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = NQ_test.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(NQ_test.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = NQ_test.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(NQ_test.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = NQ_test.index.time
                ntrans_d = ntrans.groupby(NQ_test.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect the necessary results into one object
                summary = pd.DataFrame({
                    'signalEMA': signalEMA,
                    'slowEMA': slowEMA,
                    'volat_sd': volat_sd,
                    'm': m,
                    'period': '2025',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append the results to the summary
                summary_all_breakout = pd.concat([summary_all_breakout, summary], ignore_index=True)

    # it takes a (smaller) while
    # approximately 5 minutes

# check 10 strategies with the best net_SR_mom (momentum strategies)

summary_all_breakout.sort_values(by = 'net_PnL_mom', 
                                 ascending = False).head(20) # 14K profit, different set of hyperparameters

# check 10 strategies with the best net_SR_mr (mean-reverting strategies)

summary_all_breakout.sort_values(by = 'net_PnL_mr', 
                                 ascending = False).head(20) # also pretty high profits

# summarize the results for the mom strategy
# in the form of a heatmap

# here we have four parameters
# signalEMA, slowEMA, volat_sd and m,
# so to present the results in the form of a heatmap,
# we need to combine them in pairs

summary_all_breakout["signalEMA_slowEMA"] = (
    summary_all_breakout["signalEMA"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["slowEMA"].astype(int).astype(str).str.zfill(3)
)

summary_all_breakout["volat_sd_m"] = (
    summary_all_breakout["volat_sd"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["m"].astype(str)
)

summary_all_breakout.head()

# now we can plot the heatmap
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mom',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# now we can plot the heatmap
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mr',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# Momentum: different set of hyperparameters, what about previous model with the highest net SR are for signalEMA = 20 and slowEMA = 120

# let's check the sensitivity analysis of SR to m and volat_sd

# we select only rows with values of signalEMA and slowEMA
summary_all_breakout_wybrane = summary_all_breakout[
    (summary_all_breakout['signalEMA'] == 20) & (summary_all_breakout['slowEMA'] == 120)
]

# and we create a heatmap for them
plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'net_SR_mom',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# not with all paramet3ers, but still some positive values

# heatmap plot can also be used to analyze transaction frequency

plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'av_daily_ntrans',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Average Daily Number of Transactions (Momentum strategy)',
    cmap = "Blues")

    # in line with expectations
    # the larger the m, the fewer the transactions

# #### S&P500 FUTURES (SP)


data1 = pd.read_pickle("data1_input.pkl")
data1.head(401)

# Let's split the data into training and test sets
# let's assume that data for the first two years (2023, 2024) is for training
# and data for 2025 is for testing

# train: all data up to end of 2024
data1_train = data1[data1.index < "2025-01-01"]

# test: all data from 2025 onwards
data1_test = data1[data1.index >= "2025-01-01"]

data1_train.tail()

# Lets take NQ values

SP = data1_train['SP']

# check sample break in session

plt.figure(figsize = (10,6))
plt.plot(SP[960:12000])

# Let's assume that we CLOSE ALL POSITIONS before the break (at 15:40).
# so finally, the position will always be 0 between 14:40 and 16:00.
# similarly as earlier, let's create an object named "pos_flat" 
# = 1 if position has to be flat (= 0) - we do not trade
# = 0 otherwise

# let's fill it first with zeros
pos_flat = np.zeros(len(SP))

# put our assumptions into pos_flat
pos_flat = np.zeros(len(SP))

t = SP.index.time

breaks = (
    ((t >= pd.to_datetime("09:30").time()) & (t <= pd.to_datetime("09:55").time())) |
    ((t >= pd.to_datetime("15:40").time()) & (t <= pd.to_datetime("16:00").time()))
)

pos_flat[breaks] = 1


# let's see an example break in the session from the perspective of pos_flat

plt.figure(figsize = (10,6))
plt.plot(pos_flat[960:1600])

# lets check which weekdays we have data for

dweek_ = SP.index.dayofweek + 1  # Adjust so that 0 = Monday, ..., 6 = Sunday
print(dweek_.value_counts())

# no Saturdays and Sundays in the data (perfect distribution?)

# lets create a time_ object (vector of times)

time_ = SP.index.time

# and let's fill the pos_flat vector with ones for weekends

pos_flat[((dweek_ == 5) & (time_ > pd.to_datetime('16:00').time())) |      # end of Friday
          (dweek_ == 6) |                                                  # whole Saturday (just in case)
          ((dweek_ == 7) & (time_ <= pd.to_datetime('9:55').time()))] = 1 # beginning of Sunday

# Strategy 1: 2 intersecting moving averages


# we check various parameter combinations in a loop

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

fastEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 75, 90, 120, 150, 180, 240, 300, 360, 420]

# create a dataframe to store results
summary_all_2MAs = pd.DataFrame()

# Loop over different parameter combinations
for fastEMA in fastEMA_parameters:
    for slowEMA in slowEMA_parameters:
                
                # ensure that fastEMA is less than slowEMA
                if fastEMA >= slowEMA:
                    continue

                print(f"fastEMA = {fastEMA}, slowEMA = {slowEMA}")

                # We calculate the appropriate EMA
                fastEMA_values = SP.ewm(span = fastEMA).mean()
                slowEMA_values = SP.ewm(span = slowEMA).mean()

                # Insert NaNs wherever the original price is missing
                fastEMA_values[SP.isna()] = np.nan
                slowEMA_values[SP.isna()] = np.nan 

                # Calculate position for momentum strategy
                cond2b_mom_long = fastEMA_values.shift(1) > slowEMA_values.shift(1)
                
                # let's add filters that check for the presence of NaN values
                fastEMA_nonmiss = fastEMA_values.shift(1).notna()
                slowEMA_nonmiss = slowEMA_values.shift(1).notna()

                # Now we can add these conditions to our strategies
                # if any of the values is missing,
                # we cannot make a position decision

                pos_mom = np.where(fastEMA_nonmiss & slowEMA_nonmiss,
                                   np.where(cond2b_mom_long, 1, -1),
                                   np.nan)
                pos_mr = -pos_mom 

                # Set position to 0 where pos_flat is 1
                pos_mom[pos_flat == 1] = 0
                pos_mr[pos_flat == 1] = 0
                
                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * SP.diff()), 0, pos_mom * SP.diff() * 50) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * SP.diff()), 0, pos_mr * SP.diff() * 50) 
                # point value for SP

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on SP
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on SP
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = SP.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(SP.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = SP.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(SP.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = SP.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(SP.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = SP.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(SP.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = SP.index.time
                ntrans_d = ntrans.groupby(SP.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect necessary results into one object
                summary = pd.DataFrame({
                    'fastEMA': fastEMA,
                    'slowEMA': slowEMA,
                    'period': '2023-2024',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append results to the summary
                summary_all_2MAs = pd.concat([summary_all_2MAs, summary], ignore_index=True)

# lets see top 5 stategies with respect to net_SR_mom
summary_all_2MAs.sort_values(by = 'net_PnL_mom',
                            ascending = False).head(5) #High profits and net SR ratios on momentum strategies (up to +31K $!)

# lets see top 5 stategies with respect to net_SR_mr
summary_all_2MAs.sort_values(by = 'net_PnL_mr',
                            ascending = False).head(5)

# we wil use it to visualize net SR values
# for the momentum strategy (MOM)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Momentum strategies)'
)

#Several combinations with positive net SR (90:150 the best)

# we wil use it to visualize net SR values
# for the mean-reverting strategy (MR)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Mean-reverting strategies)'
)

# No positive SR


# How about net PnL for Momentum strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Momentum Strategy)'
) #Strategy with highest net SR leads to higher profit, results seems pretty reasonable

# How about net PnL for Mean-reverting strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Mean-reverting Strategy)'
)  #No positive PnL

# Strategy 2: Volatility breakout


#Update of sharpe ratio so it won't divide by 0
def mySR(x, scale=252):
    mu = np.nanmean(x)
    sigma = np.nanstd(x)

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return np.sqrt(scale) * mu / sigma

# let's use fastEMA as the strategy signal
# and use slowEMA+/-m*std as volatility boundaries

from functions.position_VB import positionVB

# we check various parameter combinations in a loop

signalEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 90, 120, 150, 180, 200, 240, 300]
volat_sd_parameters = [60, 90, 120]
m_parameters = [1, 2, 3]

# create a dataframe to store results
summary_all_breakout = pd.DataFrame()

# loop over different parameter combinations
for signalEMA in signalEMA_parameters:
    print(f"signalEMA = {signalEMA}")
    for slowEMA in slowEMA_parameters:
        for volat_sd in volat_sd_parameters:
            for m in m_parameters:
               
                # We calculate the appropriate EMA
                signalEMA_values = SP.ewm(span = signalEMA).mean().to_numpy()
                slowEMA_values = SP.ewm(span = slowEMA).mean().to_numpy()
                                
                # We calculate the standard deviation
                volat_sd_values = SP.rolling(window = volat_sd).std().to_numpy()

                # Insert NaNs wherever the original price is missing
                signalEMA_values[SP.isna()] = np.nan
                slowEMA_values[SP.isna()] = np.nan 
                volat_sd_values[SP.isna()] = np.nan 

                # Calculate position for momentum strategy
                pos_mom = positionVB(signal = signalEMA_values, 
                                     lower = slowEMA_values - m * volat_sd_values,
                                     upper = slowEMA_values + m * volat_sd_values,
                                     pos_flat = pos_flat,
                                     strategy = "mom")
                
                pos_mr = -pos_mom 

                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * SP.diff()), 0, pos_mom * SP.diff() * 50) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * SP.diff()), 0, pos_mr * SP.diff() * 50) 
                # point value for SP = 50$

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on SP
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on SP
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = SP.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(SP.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = SP.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(SP.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = SP.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(SP.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = SP.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(SP.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = SP.index.time
                ntrans_d = ntrans.groupby(SP.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect the necessary results into one object
                summary = pd.DataFrame({
                    'signalEMA': signalEMA,
                    'slowEMA': slowEMA,
                    'volat_sd': volat_sd,
                    'm': m,
                    'period': '2023-2024',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append the results to the summary
                summary_all_breakout = pd.concat([summary_all_breakout, summary], ignore_index=True)

    # it takes a while
    # approximately 11 (!) minutes

# check 10 strategies with the best net_SR_mom (momentum strategies)

summary_all_breakout.sort_values(by = 'net_PnL_mom', 
                                 ascending = False).head(10) # 31K profit, slightly lower than for 2xEMA

# check 10 strategies with the best net_SR_mr

summary_all_breakout.sort_values(by = 'net_PnL_mr', 
                                 ascending = False).head(10) #Here also not bat, 19K net profit

# summarize the results for the mom strategy
# in the form of a heatmap

# here we have four parameters
# signalEMA, slowEMA, volat_sd and m,
# so to present the results in the form of a heatmap,
# we need to combine them in pairs

summary_all_breakout["signalEMA_slowEMA"] = (
    summary_all_breakout["signalEMA"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["slowEMA"].astype(int).astype(str).str.zfill(3)
)

summary_all_breakout["volat_sd_m"] = (
    summary_all_breakout["volat_sd"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["m"].astype(str)
)

summary_all_breakout.head()

# now we can plot the heatmap
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mom',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# for mean-reverting
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mr',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# Momentum: the highest net SR are for signalEMA = 10 and slowEMA = 60

# let's check the sensitivity analysis of SR to m and volat_sd

# we select only rows with values of signalEMA and slowEMA
summary_all_breakout_wybrane = summary_all_breakout[
    (summary_all_breakout['signalEMA'] == 10) & (summary_all_breakout['slowEMA'] == 60)
]

# and we create a heatmap for them
plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'net_SR_mom',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# Rather not an anomally

# heatmap plot can also be used to analyze transaction frequency

plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'av_daily_ntrans',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Average Daily Number of Transactions (Momentum strategy)',
    cmap = "Blues")

    # in line with expectations
    # the larger the m, the fewer the transactions

# Testing of developped strategies


# Lets take NQ values

SP_test = data1_test['SP']

# check sample break in session

plt.figure(figsize = (10,6))
plt.plot(SP[960:12000])

# put our assumptions into pos_flat
pos_flat = np.zeros(len(SP_test))

t = SP_test.index.time

breaks = (
    ((t >= pd.to_datetime("09:30").time()) & (t <= pd.to_datetime("09:55").time())) |
    ((t >= pd.to_datetime("15:40").time()) & (t <= pd.to_datetime("16:00").time()))
)

pos_flat[breaks] = 1


# let's see an example break in the session from the perspective of pos_flat

plt.figure(figsize = (10,6))
plt.plot(pos_flat[960:1600])

# lets check which weekdays we have data for

dweek_ = SP_test.index.dayofweek + 1  # Adjust so that 0 = Monday, ..., 6 = Sunday
print(dweek_.value_counts())

# no Saturdays and Sundays in the data

# lets create a time_ object (vector of times)

time_ = SP_test.index.time

# and let's fill the pos_flat vector with ones for weekends

pos_flat[((dweek_ == 5) & (time_ > pd.to_datetime('16:00').time())) |      # end of Friday
          (dweek_ == 6) |                                                  # whole Saturday (just in case)
          ((dweek_ == 7) & (time_ <= pd.to_datetime('9:55').time()))] = 1 # beginning of Sunday

# Strategy 1: 2 intersecting moving averages


# we check various parameter combinations in a loop

fastEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 75, 90, 120, 150, 180, 240, 300, 360, 420]

# create a dataframe to store results
summary_all_2MAs = pd.DataFrame()

# Loop over different parameter combinations
for fastEMA in fastEMA_parameters:
    for slowEMA in slowEMA_parameters:
                
                # ensure that fastEMA is less than slowEMA
                if fastEMA >= slowEMA:
                    continue

                print(f"fastEMA = {fastEMA}, slowEMA = {slowEMA}")

                # We calculate the appropriate EMA
                fastEMA_values = SP_test.ewm(span = fastEMA).mean()
                slowEMA_values = SP_test.ewm(span = slowEMA).mean()

                # Insert NaNs wherever the original price is missing
                fastEMA_values[SP_test.isna()] = np.nan
                slowEMA_values[SP_test.isna()] = np.nan 

                # Calculate position for momentum strategy
                cond2b_mom_long = fastEMA_values.shift(1) > slowEMA_values.shift(1)
                
                # let's add filters that check for the presence of NaN values
                fastEMA_nonmiss = fastEMA_values.shift(1).notna()
                slowEMA_nonmiss = slowEMA_values.shift(1).notna()

                # Now we can add these conditions to our strategies
                # if any of the values is missing,
                # we cannot make a position decision

                pos_mom = np.where(fastEMA_nonmiss & slowEMA_nonmiss,
                                   np.where(cond2b_mom_long, 1, -1),
                                   np.nan)
                pos_mr = -pos_mom 

                # Set position to 0 where pos_flat is 1
                pos_mom[pos_flat == 1] = 0
                pos_mr[pos_flat == 1] = 0
                
                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * SP_test.diff()), 0, pos_mom * SP_test.diff() * 50) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * SP_test.diff()), 0, pos_mr * SP_test.diff() * 50) 
                # point value for SP

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on SP
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on SP
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = SP_test.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(SP_test.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = SP_test.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(SP_test.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = SP_test.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(SP_test.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = SP_test.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(SP_test.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = SP_test.index.time
                ntrans_d = ntrans.groupby(SP_test.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect necessary results into one object
                summary = pd.DataFrame({
                    'fastEMA': fastEMA,
                    'slowEMA': slowEMA,
                    'period': '2025',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append results to the summary
                summary_all_2MAs = pd.concat([summary_all_2MAs, summary], ignore_index=True)

# lets see top 5 stategies with respect to net_SR_mom
summary_all_2MAs.sort_values(by = 'net_PnL_mom',
                            ascending = False).head(15) #net profit with negative value

# lets see top 5 stategies with respect to net_SR_mr
summary_all_2MAs.sort_values(by = 'net_PnL_mr',
                            ascending = False).head(5) #actual jump of profit up to 71K ???

# we wil use it to visualize net SR values
# for the momentum strategy (MOM)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Momentum strategies)'
)

#Some different results

# we wil use it to visualize net SR values
# for the mean-reverting strategy (MR)

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_SR_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    title = 'Net Sharpe Ratio (Mean-reverting strategies)'
)

#Mean-reverting on testing is better? inconsistent results with training though

# How about net PnL for MOM strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mom',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Momentum Strategy)'
) #Different results, bottom-right corner pretty consistent (slowEMA=420 ^ fastEMA=60 as safe pick)

# How about net PnL for MR strategy?

plot_heatmap(
    summary_all_2MAs,
    value_col = 'net_PnL_mr',
    index_col = 'fastEMA',
    columns_col = 'slowEMA',
    cmap = "coolwarm",
    title = 'Net PnL  (Mean-reverting Strategy)'
) 

# Strategy 2: Volatility breakout


# lets use fastEMA as the strategy signal
# and use slowEMA+/-m*std as volatility boundaries

#Update of sharpe ratio so it won't divide by 0
def mySR(x, scale=252):
    mu = np.nanmean(x)
    sigma = np.nanstd(x)

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return np.sqrt(scale) * mu / sigma

from functions.position_VB import positionVB

# we check various parameter combinations in a loop

signalEMA_parameters = [10, 15, 20, 30, 45, 60, 75, 90]
slowEMA_parameters = [60, 90, 120, 150, 180, 200, 240, 300]
volat_sd_parameters = [60, 90, 120]
m_parameters = [1, 2, 3]

# create a dataframe to store results
summary_all_breakout = pd.DataFrame()

# loop over different parameter combinations
for signalEMA in signalEMA_parameters:
    print(f"signalEMA = {signalEMA}")
    for slowEMA in slowEMA_parameters:
        for volat_sd in volat_sd_parameters:
            for m in m_parameters:
               
                # We calculate the appropriate EMA
                signalEMA_values = SP_test.ewm(span = signalEMA).mean().to_numpy()
                slowEMA_values = SP_test.ewm(span = slowEMA).mean().to_numpy()
                                
                # We calculate the standard deviation
                volat_sd_values = SP_test.rolling(window = volat_sd).std().to_numpy()

                # Insert NaNs wherever the original price is missing
                signalEMA_values[SP_test.isna()] = np.nan
                slowEMA_values[SP_test.isna()] = np.nan 
                volat_sd_values[SP_test.isna()] = np.nan 

                # Calculate position for momentum strategy
                pos_mom = positionVB(signal = signalEMA_values, 
                                     lower = slowEMA_values - m * volat_sd_values,
                                     upper = slowEMA_values + m * volat_sd_values,
                                     pos_flat = pos_flat,
                                     strategy = "mom")
                
                pos_mr = -pos_mom 

                # Calculate gross pnl
                pnl_gross_mom = np.where(np.isnan(pos_mom * SP_test.diff()), 0, pos_mom * SP_test.diff() * 50) 
                pnl_gross_mr = np.where(np.isnan(pos_mr * SP_test.diff()), 0, pos_mr * SP_test.diff() * 50) 
                # point value for SP = 50$

                # Calculate number of transactions
                ntrans = np.abs(np.diff(pos_mom, prepend = 0))

                # Calculate net pnl
                pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $12 per transaction on SP
                pnl_net_mr = pnl_gross_mr - ntrans * 12  # cost $12 per transaction on SP
                  
                # Aggregate to daily data
                pnl_gross_mom = pd.Series(pnl_gross_mom)
                pnl_gross_mom.index = SP_test.index.time
                pnl_gross_mom_d = pnl_gross_mom.groupby(SP_test.index.date).sum()
                pnl_gross_mr = pd.Series(pnl_gross_mr)
                pnl_gross_mr.index = SP_test.index.time
                pnl_gross_mr_d = pnl_gross_mr.groupby(SP_test.index.date).sum()

                pnl_net_mom = pd.Series(pnl_net_mom)
                pnl_net_mom.index = SP_test.index.time
                pnl_net_mom_d = pnl_net_mom.groupby(SP_test.index.date).sum()
                pnl_net_mr = pd.Series(pnl_net_mr)
                pnl_net_mr.index = SP_test.index.time
                pnl_net_mr_d = pnl_net_mr.groupby(SP_test.index.date).sum()

                ntrans = pd.Series(ntrans)
                ntrans.index = SP_test.index.time
                ntrans_d = ntrans.groupby(SP_test.index.date).sum()

                # Calculate Sharpe Ratio and PnL
                gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
                net_SR_mom = mySR(pnl_net_mom_d, scale=252)
                gross_PnL_mom = pnl_gross_mom_d.sum()
                net_PnL_mom = pnl_net_mom_d.sum()
                gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
                net_SR_mr = mySR(pnl_net_mr_d, scale=252)
                gross_PnL_mr = pnl_gross_mr_d.sum()
                net_PnL_mr = pnl_net_mr_d.sum()

                av_daily_ntrans = ntrans_d.mean()

                # Collect the necessary results into one object
                summary = pd.DataFrame({
                    'signalEMA': signalEMA,
                    'slowEMA': slowEMA,
                    'volat_sd': volat_sd,
                    'm': m,
                    'period': '2025',
                    'gross_SR_mom': gross_SR_mom,
                    'net_SR_mom': net_SR_mom,
                    'gross_PnL_mom': gross_PnL_mom,
                    'net_PnL_mom': net_PnL_mom,
                    'gross_SR_mr': gross_SR_mr,
                    'net_SR_mr': net_SR_mr,
                    'gross_PnL_mr': gross_PnL_mr,
                    'net_PnL_mr': net_PnL_mr,
                    'av_daily_ntrans': av_daily_ntrans
                }, index=[0])

                # Append the results to the summary
                summary_all_breakout = pd.concat([summary_all_breakout, summary], ignore_index=True)

    # it takes a (smaller) while
    # approximately 5 minutes

# check 10 strategies with the best net_SR_mom (momentum strategies)

summary_all_breakout.sort_values(by = 'net_PnL_mom', 
                                 ascending = False).head(20) # 14K profit, different set of hyperparameters

# check 10 strategies with the best net_SR_mr (mean-reverting strategies)

summary_all_breakout.sort_values(by = 'net_PnL_mr', 
                                 ascending = False).head(20) # also pretty high profits

# summarize the results for the mom strategy
# in the form of a heatmap

# here we have four parameters
# signalEMA, slowEMA, volat_sd and m,
# so to present the results in the form of a heatmap,
# we need to combine them in pairs

summary_all_breakout["signalEMA_slowEMA"] = (
    summary_all_breakout["signalEMA"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["slowEMA"].astype(int).astype(str).str.zfill(3)
)

summary_all_breakout["volat_sd_m"] = (
    summary_all_breakout["volat_sd"].astype(int).astype(str).str.zfill(3) + "_" +
    summary_all_breakout["m"].astype(str)
)

summary_all_breakout.head()

# now we can plot the heatmap
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mom',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# now we can plot the heatmap
plot_heatmap(
    summary_all_breakout,
    value_col = 'net_SR_mr',
    index_col = 'signalEMA_slowEMA',
    columns_col = 'volat_sd_m',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# let's check the sensitivity analysis of SR to m and volat_sd

# we select only rows with values of signalEMA and slowEMA
summary_all_breakout_wybrane = summary_all_breakout[
    (summary_all_breakout['signalEMA'] == 20) & (summary_all_breakout['slowEMA'] == 150)
]

# and we create a heatmap for them
plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'net_SR_mom',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Net Sharpe Ratio (Momentum strategy)',
    cmap = "coolwarm"
)

# not with all paramet3ers, but still some positive values

# heatmap plot can also be used to analyze transaction frequency

plot_heatmap(
    summary_all_breakout_wybrane,
    value_col = 'av_daily_ntrans',
    index_col = 'volat_sd',
    columns_col = 'm',
    title = 'Average Daily Number of Transactions (Momentum strategy)',
    cmap = "Blues")

    # in line with expectations
    # the larger the m, the fewer the transactions

# ## PAIR TRADING STRATEGIES 


data1 = pd.read_pickle("data1_input.pkl")
data1.head(401)

# based on closing prices
# we will calculate logarithmic rates of return for
# all series in basis points (bps) 1bps = 0.01% = 0.0001,
# to obtain basis points of return, multiply by 10,000

# Shift index by desired number of periods - can be positive or negative
data1_r = np.log(data1 / data1.shift(1)) * 10000

# changing the column names to make them appropriate
# (replace close_ with r_ )

data1_r.columns = ["r_NQ", "r_SP"]

data1_r.head(401)

# lets create an object with selected columns (needed for analyses)
# only closing prices and rates of return for NQ and SP

dataNQ_SP = pd.concat(
    [data1[['NQ', 'SP']], 
     data1_r[['r_NQ', 'r_SP']]],
    axis=1
)

dataNQ_SP.head(401)

# lets see a figure of all columns
# with closing prices and returns for NQ and SP

# We don't want to treat time as a continuous variable in the graph
# (because there are "holes" from 4:00 PM to 9:30 AM the next day)

# Let's convert the datetime index to a text index and pass it as the x-axis
dataNQ_SP_plot = dataNQ_SP.copy()
dataNQ_SP_plot['time'] = dataNQ_SP_plot.index.astype(str)

# We reset the index to make 'time' a column
dataNQ_SP_plot = dataNQ_SP_plot.reset_index(drop = True)

# Let's set time as the X-axis and draw the graphs
dataNQ_SP_plot.plot(
    x = 'time',
    subplots = True,
    layout = (2, 2),
    title = "Quotations of NQ and SP",
    figsize = (12, 10)
)
plt.show()

dataNQ_SP.isna().sum()

# Spread based on indices quotations
# 
# lets formulate a spread: `P1 - m * P2` (here `SP - m * NQ`)
# 
# where `m = m1/m2` is based on average ratio between the prices on the **PREVIOUS** day
# 
# Spread is a signal to our model, which shows whether to take position or not (volatility bands around the spread).
# 
# **CAUTION**! we assume the mean reverting behavior of the spread! 
# 
# We will calculate `m` based on **previous day's average prices**


# lets calculate average ratio of prices on the daily basis

# Compute the ratio SP / NQ
ratio = dataNQ_SP["SP"] / dataNQ_SP["NQ"]

# Compute daily averages
US_av_ratio = ratio.resample("D").mean()
# keep non-missing values only
US_av_ratio = US_av_ratio.dropna()

# Assign the name
US_av_ratio = US_av_ratio.to_frame(name="av_ratio")

# Plot how the average ratio looks like
plt.plot(US_av_ratio.index, 
         US_av_ratio["av_ratio"])

# between 0.27 and 0.35


US_av_ratio.head()

# check first 10 dates
US_av_ratio.index[:10]

# lets move it to 9:41 AM of the next day
US_av_ratio.index[:10] + pd.Timedelta("1D") + pd.Timedelta("9h41m")

US_av_ratio.index[:10].day_name() #no Saturdays nor Sundays

# lets use the above information to adjust the timestamps
US_av_ratio.index[:10] + pd.to_timedelta(np.where(US_av_ratio.index[:10].day_name() == "Friday", "3D", "1D")) + pd.Timedelta("9h41m")

# now it looks good

# lets apply the changes in our data object
US_av_ratio.index = US_av_ratio.index + pd.to_timedelta(np.where(US_av_ratio.index.day_name() == "Friday", "3D", "1D")) + pd.Timedelta("9h41m")

US_av_ratio.head()

# Spread based on returns
# 
# alternative spread based on **RETURNS**:
# 
# `r1 - ms * r2` (here `SP - m * NQ`)
# 
# where `ms = s1/s2` is based on the ratio of **standard deviations** of returns on the **PREVIOUS** day


dataNQ_SP.head(401)

# Daily standard deviation ratio
daily_std = (
    dataNQ_SP
    .resample("D")
    .agg({
        "r_SP": "std",
        "r_NQ": "std"
    })
)

US_sds_ratio = (
    daily_std["r_SP"] / daily_std["r_NQ"]
).to_frame(name="sds_ratio")

US_sds_ratio = US_sds_ratio.replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(10, 5))
plt.plot(US_sds_ratio.index, US_sds_ratio["sds_ratio"])
plt.title("Daily Std Ratio: SP / NQ")
plt.xlabel("Date")
plt.ylabel("Std Ratio")
plt.grid(True)
plt.show()


# lets move the index to 9:31 of the next trading day

US_sds_ratio.index = US_sds_ratio.index + pd.to_timedelta(np.where(US_sds_ratio.index.day_name() == "Friday", "3D", "1D")) + pd.Timedelta("9h41m")

# we need to merge our basic 1 min data with daily calculations

dataUSA_2 = dataNQ_SP.copy()
dataUSA_2 = dataUSA_2.merge(US_av_ratio, 
                            # we want to use indexes 
                            # as merging keys
                            left_index = True, 
                            right_index = True, 
                            how = "left")
dataUSA_2 = dataUSA_2.merge(US_sds_ratio, 
                            left_index = True, 
                            right_index = True, 
                            how = "left")

# lets see how it worked

dataUSA_2.between_time("09:41", "09:55").head(30)

# there are a lot of missings in a the last 2 columns
# which should be filled with the last non-missing value
# (last multiplier is used until there is a new one)

# We apply forward fill method to the last two columns
dataUSA_2[["av_ratio", "sds_ratio"]] = dataUSA_2[["av_ratio", "sds_ratio"]].ffill()

# and check the results
dataUSA_2.between_time("09:41", "09:55").head(30)

# now the values are filled properly

# lets make sure that we exclude weekends from our data

dataUSA_2.index.day_name().value_counts()

# there are no weekends in the data

# now we can calculate the spread (in 2 variants)
dataUSA_2["spread_avratio"] = dataUSA_2["SP"] - dataUSA_2["av_ratio"] * dataUSA_2["NQ"]
dataUSA_2["spread_sdsratio"] = dataUSA_2["r_SP"] - dataUSA_2["sds_ratio"] * dataUSA_2["r_NQ"]

# plot both spreads in separate subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

axes[0].plot(dataUSA_2.index, 
             dataUSA_2["spread_avratio"], 
             label = "Spread (av_ratio)")
axes[0].set_title("Spread (Average Ratio)")
axes[0].legend()

axes[1].plot(dataUSA_2.index, 
             dataUSA_2["spread_sdsratio"], 
             label = "Spread (sds_ratio)")
axes[1].set_title("Spread (Standard Deviation Ratio)")
axes[1].legend()

plt.tight_layout()
plt.show()

# we assume that spread mean reverts to 0

# lets apply the volatility breakout model

# first we need to calculate the standard deviation of the spreads
# let's use rolling window of 120 minutes (2 hours)

dataUSA_2["std_spread_avratio"] = dataUSA_2["spread_avratio"].rolling(window=120).std()
dataUSA_2["std_spread_sdsratio"] = dataUSA_2["spread_sdsratio"].rolling(window=120).std()

# lets put missings whenever SP price is missing
# (KO price should be missing in the same moments)

dataUSA_2.loc[dataUSA_2["SP"].isna(), ["std_spread_avratio", "std_spread_sdsratio"]] = np.nan


# applying a volatility breakout model
# sample upper and lower bounds for spreads
# for a volatility multiplier of 3
# (here we put the upper and lower band along zero)

dataUSA_2["upper_bound_avratio"] = 3 * dataUSA_2["std_spread_avratio"]
dataUSA_2["lower_bound_avratio"] = -3 * dataUSA_2["std_spread_avratio"]

# lets see how it looks like
# ignoring time dimension for clarity

dataUSA_2_plot = dataUSA_2.reset_index()
plt.figure(figsize=(12, 6))
plt.plot(dataUSA_2_plot.index, 
         dataUSA_2_plot["spread_avratio"], 
         label="Spread (av_ratio)")
plt.plot(dataUSA_2_plot.index, 
         dataUSA_2_plot["upper_bound_avratio"], 
         label="Upper Bound", linestyle='--')
plt.plot(dataUSA_2_plot.index, 
         dataUSA_2_plot["lower_bound_avratio"], 
         label="Lower Bound", linestyle='--')
plt.title("Spread with Upper and Lower Bounds (av_ratio)")
plt.legend()
plt.show()  

# the same for spread_sdsratio

dataUSA_2["upper_bound_sdsratio"] = 3 * dataUSA_2["std_spread_sdsratio"]
dataUSA_2["lower_bound_sdsratio"] = -3 * dataUSA_2["std_spread_sdsratio"]

# lets see how it looks like
# ignoring time dimension for clarity
dataUSA_2_plot = dataUSA_2.reset_index()
plt.figure(figsize=(12, 6))
plt.plot(dataUSA_2_plot.index, 
         dataUSA_2_plot["spread_sdsratio"], 
         label="Spread (sds_ratio)")
plt.plot(dataUSA_2_plot.index, 
         dataUSA_2_plot["upper_bound_sdsratio"],
         label="Upper Bound", linestyle='--')
plt.plot(dataUSA_2_plot.index, 
         dataUSA_2_plot["lower_bound_sdsratio"],
         label="Lower Bound", linestyle='--')
plt.title("Spread with Upper and Lower Bounds (sds_ratio)")
plt.legend()
plt.show()

### position will be based on relation of the spread to volatility bands

# lets assume we do not trade within the first 25-mins of the day
# and exit all positions 20 minutes before the end of quotations

# lets create a pos_flat vector and fill it with 0s

pos_flat = np.zeros(len(dataUSA_2))

# we do not trade within the first quarter (9:31-9:45) 
# but also before that time since midnight

pos_flat[dataUSA_2.index.time <= pd.to_datetime("9:55").time()] = 1

# and last quarter of the session (15:46-16:00)
# but also after this time until midnight

pos_flat[dataUSA_2.index.time >= pd.to_datetime("15:41").time()] = 1

# !!! there are no weekends in our data, so we do not need 
# to control for that in pos_flat

pd.Series(pos_flat).value_counts()

# lets use the positionVB() function known from previous labs
# to calculate the position based on spread_avratio

from functions.position_VB import positionVB

dataUSA_2["pos_avratio"] = positionVB(signal = dataUSA_2["spread_avratio"],
                                      lower = dataUSA_2["lower_bound_avratio"],
                                      upper = dataUSA_2["upper_bound_avratio"],
                                      pos_flat = pos_flat,
                                      # IMPORTANT !!!!
                                      strategy = "mr")

dataUSA_2["pos_avratio"].value_counts() #a lot of passive 'hold' positions

# lets create a vector of the number of transactions

dataUSA_2["n_trans_avratio"] = np.abs(np.diff(dataUSA_2["pos_avratio"], prepend = 0))

# next we calculate the gross P&L from the strategy
# for every minute 
# we multiply the position by the return of the spread

# gross PnL = position * (dP_SP - ratio * dP_NQ) 

dataUSA_2["pnl_gross_avratio"] = dataUSA_2["pos_avratio"] * (dataUSA_2["SP"].diff() - dataUSA_2["av_ratio"] * dataUSA_2["NQ"].diff())

#Trading costs assumptions

trcost_SP = round(data1["SP"].mean() * 0.00235, 2)
trcost_NQ = round(data1["NQ"].mean() * 0.0006798, 2) 

print(f"Trading cost per transaction for SP: {trcost_SP:.2f} USD") #should be 12$
print(f"Trading cost per transaction for NQ: {trcost_NQ:.2f} USD")

# pnl after  costs

# !!! REMEMBER that we trade one unit of KO and av_ratio units of PEP
# AND there is NO minus "-" in the costs - they are always positive !!!

dataUSA_2["pnl_net_avratio"] = dataUSA_2["pnl_gross_avratio"] - dataUSA_2["n_trans_avratio"] * (trcost_SP + dataUSA_2["av_ratio"] * trcost_NQ)

# lets calculate and plot cumulative gross PnL
# without adding a new column to the data

dataUSA_2["pnl_gross_avratio"].cumsum().plot(figsize=(12,6), 
                                             title="Cumulative Gross PnL (av_ratio)")
plt.show()

# positive, still some NA's

# what about cummulative net PnL?

dataUSA_2["pnl_net_avratio"].cumsum().plot(figsize=(12,6), 
                                           title="Cumulative Net PnL (av_ratio)")
plt.show()

# big loss

# lets do a comparison within a loop for spread_avratio and spread_sdsratio (GridSearch)

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

volat_sd_parameters = [60, 90, 120, 150, 180]
m_parameters = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

# create a dataframe to store results
summary_all_NQ_SP = pd.DataFrame()

for volat_sd in volat_sd_parameters:
    for m in m_parameters:
        print(f"volat_sd: {volat_sd}, m: {m}")

        # calculate teh elements of the strategy
        SP = dataUSA_2["SP"]
        NQ = dataUSA_2["NQ"]

        # spread based on average ratio
        signal_avratio = SP - (dataUSA_2["av_ratio"] * NQ)
        std_spread_avratio = signal_avratio.rolling(window=volat_sd).std()
        upper_bound_avratio = m * std_spread_avratio
        lower_bound_avratio = -m * std_spread_avratio
        # position
        pos_avratio = positionVB(signal = signal_avratio,
                                lower = lower_bound_avratio,
                                upper = upper_bound_avratio,
                                pos_flat = pos_flat,
                                strategy = "mr")
        # number of transactions
        n_trans_avratio = np.abs(np.diff(pos_avratio, prepend = 0))
        # convert to pd.Series and set the index
        n_trans_avratio = pd.Series(n_trans_avratio, index=dataUSA_2.index)

        # gross and net PnL
        pnl_gross_avratio = pos_avratio * (dataUSA_2["SP"].diff() - dataUSA_2["av_ratio"] * dataUSA_2["NQ"].diff())
        pnl_net_avratio = pnl_gross_avratio - n_trans_avratio * (trcost_SP + dataUSA_2["av_ratio"] * trcost_NQ)

        # spread based on standard deviation ratio
        signal_sdsratio = dataUSA_2["spread_sdsratio"]
        std_spread_sdsratio = signal_sdsratio.rolling(window=volat_sd).std()
        upper_bound_sdsratio = m * std_spread_sdsratio
        lower_bound_sdsratio = -m * std_spread_sdsratio
        # position
        pos_sdsratio = positionVB(signal = signal_sdsratio,
                                lower = lower_bound_sdsratio,
                                upper = upper_bound_sdsratio,
                                pos_flat = pos_flat,
                                strategy = "mr")
        
        # number of transactions
        n_trans_sdsratio = np.abs(np.diff(pos_sdsratio, prepend = 0))
        # convert to pd.Series and set the index
        n_trans_sdsratio = pd.Series(n_trans_sdsratio, index=dataUSA_2.index)

        # !!!!! signal is based on returns, but PnL on prices !!!!
        pnl_gross_sdsratio = pos_sdsratio * (dataUSA_2["SP"].diff() - dataUSA_2["sds_ratio"] * dataUSA_2["NQ"].diff())
        pnl_net_sdsratio = pnl_gross_sdsratio - n_trans_sdsratio * (trcost_SP + dataUSA_2["sds_ratio"] * trcost_NQ)

        # aggregate to daily
        pnl_gross_avratio_daily = pnl_gross_avratio.resample("D").sum().dropna()
        pnl_gross_sdsratio_daily = pnl_gross_sdsratio.resample("D").sum().dropna()
        pnl_net_avratio_daily = pnl_net_avratio.resample("D").sum().dropna()
        pnl_net_sdsratio_daily = pnl_net_sdsratio.resample("D").sum().dropna()
        n_trans_avratio_daily = n_trans_avratio.resample("D").sum().dropna()
        n_trans_sdsratio_daily = n_trans_sdsratio.resample("D").sum().dropna()

        # calculate summary measures
        gross_SR_avratio = mySR(pnl_gross_avratio_daily, scale = 252)
        net_SR_avratio = mySR(pnl_net_avratio_daily, scale = 252)
        gross_PnL_avratio = pnl_gross_avratio_daily.sum()
        net_PnL_avratio = pnl_net_avratio_daily.sum()
        av_daily_ntrans_avratio = n_trans_avratio_daily.mean()
        
        gross_SR_sdsratio = mySR(pnl_gross_sdsratio_daily, scale = 252)
        net_SR_sdsratio = mySR(pnl_net_sdsratio_daily, scale = 252)
        gross_PnL_sdsratio = pnl_gross_sdsratio_daily.sum()
        net_PnL_sdsratio = pnl_net_sdsratio_daily.sum()
        av_daily_ntrans_sdsratio = n_trans_sdsratio_daily.mean()
        
        # Collect the necessary results into one object
        summary = pd.DataFrame({
                'volat_sd': volat_sd,
                'm': m,
                'gross_SR_avratio': gross_SR_avratio,
                'net_SR_avratio': net_SR_avratio,
                'gross_PnL_avratio': gross_PnL_avratio,
                'net_PnL_avratio': net_PnL_avratio,
                'av_daily_ntrans_avratio': av_daily_ntrans_avratio,
                'gross_SR_sdsratio': gross_SR_sdsratio,
                'net_SR_sdsratio': net_SR_sdsratio,
                'gross_PnL_sdsratio': gross_PnL_sdsratio,
                'net_PnL_sdsratio': net_PnL_sdsratio,
                'av_daily_ntrans_sdsratio': av_daily_ntrans_sdsratio
                }, index=[0])

        # Append the results to the summary
        summary_all_NQ_SP = pd.concat([summary_all_NQ_SP, 
                                        summary], 
                                    ignore_index = True)

# it takes about 10 minutes to run

# lets import a function written by the lecturer
# to visualize the strategy results in a form of a heatmap
from functions.plot_heatmap import plot_heatmap

# and see the results on the heatmap graph

# gross SR - spread av_ratio
plot_heatmap(summary_all_NQ_SP, 
             value_col = "gross_SR_avratio", 
             index_col = "volat_sd", 
             columns_col = "m", 
             title = "Gross SR (av_ratio) for SP-NQ Strategy")

# net SR - spread av_ratio
plot_heatmap(summary_all_NQ_SP,
             value_col = "net_SR_avratio", 
             index_col = "volat_sd", 
             columns_col = "m", 
             title = "Net SR (av_ratio) for SP-NQ Strategy") #well ...

# net Pnl - spread av_ratio
plot_heatmap(summary_all_NQ_SP,
             value_col = "net_PnL_avratio", 
             index_col = "volat_sd", 
             columns_col = "m", 
             title = "Net PnL (av_ratio) for SP-NQ Strategy")

# average daily ntrans

plot_heatmap(summary_all_NQ_SP,
             value_col = "av_daily_ntrans_avratio",
             index_col = "volat_sd",
             columns_col = "m",
             title = "Average Daily Number of Transactions (av_ratio) for SP-NQ Strategy")

# check 10 strategies with the best net_PnL_avratio

summary_all_NQ_SP.sort_values(by = 'net_PnL_avratio', 
                                 ascending = False).head(20) # also pretty high profits

# check 10 strategies with the best net_PnL_sdsratio

summary_all_NQ_SP.sort_values(by = 'net_PnL_sdsratio', 
                                 ascending = False).head(20) # also pretty high profits

# here we assume that we trade every day

# lets save the results for the next labs, when we will
# apply some additional filtering rules

dataUSA_2.to_parquet("data_pair_SP_NQ.parquet")
summary_all_NQ_SP.to_parquet("summary_pair_SP_NQ.parquet")


# Adding additional filtering to pair strategy


# 1) Correlation based filtering


# Lets calculate the correlation between KO and PEP 
# closing prices and returns for the whole sample

correlation_p = dataUSA_2['SP'].corr(dataUSA_2['NQ'])
correlation_r = dataUSA_2['r_SP'].corr(dataUSA_2['r_NQ'])

print("SP and NQ closing price correlation:")
print(correlation_p)

print("SP and PENQP returns correlation:")
print(correlation_r)

# correlation between prices is positive 
# while between returns with lower value

# lets check it on a daily basis

correlation_p_daily = dataUSA_2.resample("D").apply(lambda x: x['SP'].corr(x['NQ']))
correlation_r_daily = dataUSA_2.resample("D").apply(lambda x: x['r_SP'].corr(x['r_NQ']))

# and drop NaN values
correlation_p_daily = correlation_p_daily.dropna()
correlation_r_daily = correlation_r_daily.dropna()

# Plot daily correlation between SP and NQ prices

correlation_p_daily.plot(title="Daily correlation between SP and NQ prices")

# it is positive on most days!

# lets check how often daily correlation between prices 
# is above 0.6, 0.7, 0.8 and 0.9

print("Share of days with correlation above 0.6:",  #here the majority
      (correlation_p_daily > 0.6).sum() / len(correlation_p_daily))

print("Share of days with correlation above 0.7:", 
      (correlation_p_daily > 0.7).sum() / len(correlation_p_daily))

print("Share of days with correlation above 0.8:", 
      (correlation_p_daily > 0.8).sum() / len(correlation_p_daily))

print("Share of days with correlation above 0.9:", 
      (correlation_p_daily > 0.9).sum() / len(correlation_p_daily))


# the same for correlation between returns

correlation_r_daily.plot(title="Daily correlation between SP and NQ returns")

# check three thresholds for returns correlation
# (it is never above 0.9)
print("Share of days with returns correlation above 0.6:", 
      (correlation_r_daily > 0.6).sum() / len(correlation_r_daily))
print("Share of days with returns correlation above 0.7:", 
      (correlation_r_daily > 0.7).sum() / len(correlation_r_daily))
print("Share of days with returns correlation above 0.8:", 
      (correlation_r_daily > 0.8).sum() / len(correlation_r_daily)) #still high fraction

# lets combine daily correlations
# of prices and returns into a dataframe

daily_correlations = pd.DataFrame({
    "correlation_prices": correlation_p_daily,
    "correlation_returns": correlation_r_daily
})

# Regression based filtering


#Hypothesis from previous analyses: linear may be quite beneficial

import statsmodels.api as sm

# the OLS regression requires dropping NaN values

dataUSA_2_nonan = dataUSA_2.dropna(subset=["SP", "NQ", "r_SP", "r_NQ"])

X = dataUSA_2_nonan["NQ"]
y = dataUSA_2_nonan["SP"]
X = sm.add_constant(X)  # adding a constant term for intercept
model_ols = sm.OLS(y, X).fit()

print(model_ols.summary())

# the coefficient is positive (0,2269)
# R2 = 0.976 (!)

# lets check it on a daily basis

# we define a function performing regression for a given 
# dataframe and returning relevant statistics

import numpy as np
import pandas as pd
import statsmodels.api as sm

def regression_selected(df, y_col, x_col, add_const=True):

    # If the group is empty return NaNs
    if df.empty:
        return pd.Series({
            'beta': np.nan,
            'pvalue': np.nan,
            'tstat': np.nan,
            'r2': np.nan
        })

    y = df[y_col]
    X = df[[x_col]]  # keep as DataFrame

    if add_const:
        X = sm.add_constant(X)
        x_param_name = x_col      # slope name in params/pvalues/etc.
    else:
        x_param_name = x_col      # only column present

    model = sm.OLS(y, X).fit()

    return pd.Series({
        'beta':   model.params[x_param_name],
        'pvalue': model.pvalues[x_param_name],
        'tstat':  model.tvalues[x_param_name],
        'r2':     model.rsquared
    })


# lets check how it works on a full dataframe

regression_selected(dataUSA_2_nonan, y_col="SP", x_col="NQ")

# and apply it to daily resampled data
daily_regressions_P = dataUSA_2_nonan.resample("D").apply(
    lambda g: regression_selected(g, y_col="SP", x_col="NQ"))

daily_regressions_P.head()
# resample("D") creates groups for all calendar days, 
# including weekends/holidays where we have no data at all
# (e.g. 2025-01-04, 2025-01-05)
# that is why we had to control for empty groups in the function above

# store similar results for returns regression

daily_regressions_R = dataUSA_2_nonan.resample("D").apply(
    lambda g: regression_selected(g, y_col="r_SP", x_col="r_NQ"))

daily_regressions_R.head()

# remove rows with NaN values from daily statistics

daily_regressions_P = daily_regressions_P.dropna()
daily_regressions_R = daily_regressions_R.dropna()

# Cointegration based filtering


# lets use PP and KPSS tests for cointegration
# based on the functions from lab03
# extended to control for empty dataframes

from arch.unitroot import PhillipsPerron, KPSS

# the function to get residuals from OLS regression
def _eg_residuals(df: pd.DataFrame, col1: str, col2: str):
    X = sm.add_constant(df[col1].values, has_constant="add")
    y = df[col2].values
    model = sm.OLS(y, X).fit()
    return model.resid

# the function for PP p-value
def eg_pp_pvalue(df: pd.DataFrame, col1: str, col2: str, 
                 trend: str = "c") -> float:
    # If the group is empty return NaN
    if df.empty:
        return np.nan

    resid = _eg_residuals(df, col1, col2)
    return float(PhillipsPerron(resid, trend=trend).pvalue)

# the function for KPSS p-value
def eg_kpss_pvalue(df: pd.DataFrame, col1: str, col2: str, 
                   trend: str = "c") -> float:
    # If the group is empty return NaN
    if df.empty:
        return np.nan
    resid = _eg_residuals(df, col1, col2)
    return float(KPSS(resid, trend=trend).pvalue)

# check cointegration for the whole sample

print("p-value of PP test:", eg_pp_pvalue(dataUSA_2_nonan, "SP", "NQ"))
print("p-value of KPSS test:", eg_kpss_pvalue(dataUSA_2_nonan, "SP", "NQ"))

# lack of cointegration (PP test: very close to 5% level)

# lets combine together daily statistics for correlations
# and regression for prices and returns

# we can merge dataframes using pd.concat

combined_daily_stats = pd.concat([
    daily_correlations,
    daily_regressions_P.add_prefix("regression_price_"),
    daily_regressions_R.add_prefix("regression_return_")
], axis=1)

combined_daily_stats.head()

# but calculations based on the first day
# will be used to filter on the second day, etc.

# lets adjust the dataset accordingly
# by moving the time index to the next day

# lets do it similarly as last time
# (without changing the time to 9:31 AM)

combined_daily_stats.index = combined_daily_stats.index + pd.to_timedelta(np.where(combined_daily_stats.index.day_name() == "Friday", "3D", "1D")) 

combined_daily_stats.head()

# some summary statistics
combined_daily_stats.describe()

# based on the combined daily statistics
# lets create filtering rules
# storing them as new columns, where
# 1 means that the filter condition is met (trade on that day)
# and 0 means that it is not met (do NOT trade on that day)

# based on correlation between prices
combined_daily_stats['filter_correlation_prices_06'] = (combined_daily_stats['correlation_prices'] > 0.6) * 1
combined_daily_stats['filter_correlation_prices_07'] = (combined_daily_stats['correlation_prices'] > 0.7) * 1
combined_daily_stats['filter_correlation_prices_08'] = (combined_daily_stats['correlation_prices'] > 0.8) * 1
combined_daily_stats['filter_correlation_prices_09'] = (combined_daily_stats['correlation_prices'] > 0.9) * 1
# based on correlation between returns
combined_daily_stats['filter_correlation_returns_06'] = (combined_daily_stats['correlation_returns'] > 0.6) * 1
combined_daily_stats['filter_correlation_returns_07'] = (combined_daily_stats['correlation_returns'] > 0.7) * 1
combined_daily_stats['filter_correlation_returns_08'] = (combined_daily_stats['correlation_returns'] > 0.8) * 1
# based on regression for prices - significant beta above some threshold
combined_daily_stats['filter_regression_price_beta_sig_above0'] = ((combined_daily_stats['regression_price_beta'] > 0) & (combined_daily_stats['regression_price_pvalue'] < 0.05)) * 1
combined_daily_stats['filter_regression_price_beta_sig_above025'] = ((combined_daily_stats['regression_price_beta'] > 0.25) & (combined_daily_stats['regression_price_pvalue'] < 0.05)) * 1
combined_daily_stats['filter_regression_price_beta_sig_above05'] = ((combined_daily_stats['regression_price_beta'] > 0.5) & (combined_daily_stats['regression_price_pvalue'] < 0.05)) * 1
# based on regression for returns - significant beta above some threshold
combined_daily_stats['filter_regression_return_beta_sig_above0'] = ((combined_daily_stats['regression_return_beta'] > 0) & (combined_daily_stats['regression_return_pvalue'] < 0.05)) * 1
combined_daily_stats['filter_regression_return_beta_sig_above025'] = ((combined_daily_stats['regression_return_beta'] > 0.25) & (combined_daily_stats['regression_return_pvalue'] < 0.05)) * 1
combined_daily_stats['filter_regression_return_beta_sig_above05'] = ((combined_daily_stats['regression_return_beta'] > 0.5) & (combined_daily_stats['regression_return_pvalue'] < 0.05)) * 1

# lets consider a mean reverting strategy based 
# on last week's spread with a specific set of parameters

# to check the influence of various filters
# we will simply refer to daily pnls that can be calculated
# based on the dataUSA_2_nonan dataframe
# and its columns pnl_gross_avratio and pnl_net_avratio

daily_pnls = dataUSA_2_nonan.resample("D").agg({
    "pnl_gross_avratio": "sum",
    "pnl_net_avratio": "sum",
    "n_trans_avratio": "sum"
})

# remove weekends with no trading
daily_pnls = daily_pnls[daily_pnls.index.dayofweek < 5]

daily_pnls.head()

# combine daily pnls with daily filtering rules
# (only columns which start with 'filter_')

daily_pnls_filters = pd.concat([daily_pnls, 
                                combined_daily_stats.filter(regex='^filter_')], axis=1)

daily_pnls_filters.head()

# lets remind the profitability of the strategy without any filters

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

print("Gross PnL SR:", mySR(daily_pnls['pnl_gross_avratio'], scale = 252))
print("Net PnL SR:", mySR(daily_pnls['pnl_net_avratio'], scale = 252))

# and average number of trades per day
print("Average number of trades per day:", daily_pnls['n_trans_avratio'].mean())

# lets check the first filtering rule:
# whenever filter_correlation_prices_06 is 0 
# (remember - based on the PREVIOUS day!)
# we do not trade on that day, so we set pnl to 0

pnl_gross_avratio_filtered = daily_pnls_filters['pnl_gross_avratio'] * daily_pnls_filters['filter_correlation_prices_06']
pnl_net_avratio_filtered = daily_pnls_filters['pnl_net_avratio'] * daily_pnls_filters['filter_correlation_prices_06']

print("Gross PnL SR after applying filter_correlation_prices_06:", 
      mySR(pnl_gross_avratio_filtered, scale = 252))
print("Net PnL SR after applying filter_correlation_prices_06:", 
      mySR(pnl_net_avratio_filtered, scale = 252))

# and average number of trades per day
n_trades_filtered = daily_pnls_filters['n_trans_avratio'] * daily_pnls_filters['filter_correlation_prices_06']
print("Average number of trades per day after applying filter_correlation_prices_06:", n_trades_filtered.mean())

# Worse results

# lets compare all filtering rules
# and store the results in a dataframe

results_filters = []
for col in daily_pnls_filters.columns:
    if col.startswith('filter_'):
        pnl_gross_filtered = daily_pnls_filters['pnl_gross_avratio'] * daily_pnls_filters[col]
        pnl_net_filtered = daily_pnls_filters['pnl_net_avratio'] * daily_pnls_filters[col]
        n_trades_filtered = daily_pnls_filters['n_trans_avratio'] * daily_pnls_filters[col]
        results_filters.append({
            'filter': col,
            'gross_SR': mySR(pnl_gross_filtered, scale = 252),
            'net_SR': mySR(pnl_net_filtered, scale = 252),
            'gross_PnL': pnl_gross_filtered.sum(),
            'net_PnL': pnl_net_filtered.sum(),
            'avg_n_trades_per_day': n_trades_filtered.mean()
        })

results_filters_df = pd.DataFrame(results_filters)


# any positive net SR?
results_filters_df.sort_values(by = 'net_SR',
                               ascending = False)

# No, but close to 0 (regression-based filtering)! But with very few trades...

# ## FINAL STARTEGY APPLICATION AND ANALYSIS


#Gotowiec od Wójcika, tu wsadzasz najlepszy model i przepuszczasz

# Simple Strategy proposal #1 (single assets strategy): 
# momentum based on volatility breakout model
# NQ Futures strategy hyperparameters: fast EMA10, slow EMA120, volat_std=120, m=2
# SP Futures strategy hyperparameters:fast EMA20, slow EMA150, volat_std=120, m=1

# Loading the data

quarters = ['2023_Q1', '2023_Q3', '2023_Q4',
            '2024_Q2', '2024_Q4',
            '2025_Q1', '2025_Q2']

# create an empty DataFrame to store summary for all quarters
summary_data1_all_quarters = pd.DataFrame()

# Additional local functions

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

#Application of customized function
def run_volatility_breakout(
    price,
    fast_span,
    slow_span,
    vol_window,
    m,
    pos_flat,
    point_value
):
    # EMAs
    fast = price.ewm(span=fast_span).mean()
    slow = price.ewm(span=slow_span).mean()

    # Volatility
    vol = price.rolling(vol_window).std()

    # NaN handling
    fast[price.isna()] = np.nan
    slow[price.isna()] = np.nan
    vol[price.isna()] = np.nan

    # Positions
    pos = positionVB(
        signal=fast.to_numpy(),
        lower=(slow - m * vol).to_numpy(),
        upper=(slow + m * vol).to_numpy(),
        pos_flat=pos_flat,
        strategy="mom"
    )

    # PnL
    pnl_gross = np.where(
        np.isnan(pos * price.diff()),
        0,
        pos * price.diff() * point_value
    )

    ntrans = np.abs(np.diff(pos, prepend=0))
    pnl_net = pnl_gross - ntrans * 12

    return pnl_gross, pnl_net, ntrans

#Plotting net PnL for entire data
def plot_total_net_pnl(pnl_net_dict, symbol):
    pnl_all = pd.concat(pnl_net_dict.values()).sort_index()

    plt.figure(figsize=(12, 6))
    plt.plot(pnl_all.cumsum(), linewidth=2)

    plt.title(f'Cumulative Net PnL – {symbol} (All Quarters)')
    plt.xlabel('Date')
    plt.ylabel('PnL [$]')
    plt.grid(True)

    plt.show()

#Plotting net PnL for quarter-to-quarter
def plot_quarter_pnl(pnl_gross_d, pnl_net_d, quarter, symbol, save=True):
    plt.figure(figsize=(12, 6))

    plt.plot(
        pnl_gross_d.cumsum(),
        label="Gross PnL",
        linewidth=2
    )
    plt.plot(
        pnl_net_d.cumsum(),
        label="Net PnL",
        linewidth=2
    )

    plt.title(f"Cumulative Gross & Net PnL – {symbol} ({quarter})")
    plt.xlabel("Date")
    plt.ylabel("PnL [$]")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(
            f"data1_{quarter}_{symbol}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()

#Defining position of strategy in breakout model
def volatility_breakout_position(signal, slow, vol, m, pos_flat):
    upper = slow + m * vol
    lower = slow - m * vol

    sig_lag = signal.shift(1)
    upper_lag = upper.shift(1)
    lower_lag = lower.shift(1)

    pos = np.where(
        sig_lag.notna() & upper_lag.notna() & lower_lag.notna(),
        np.where(sig_lag > upper_lag, 1,
        np.where(sig_lag < lower_lag, -1, 0)),
        np.nan
    )

    pos[pos_flat == 1] = 0
    return pos

#Computing proper metrics for evaluation
def compute_metrics(pnl_gross, pnl_net, price_series, ntrans, scale=252):
    idx = price_series.index

    pnl_gross_d = (
        pd.Series(pnl_gross, index=idx)
        .groupby(idx.date)
        .sum()
    )

    pnl_net_d = (
        pd.Series(pnl_net, index=idx)
        .groupby(idx.date)
        .sum()
    )

    ntrans_d = (
        pd.Series(ntrans, index=idx)
        .groupby(idx.date)
        .sum()
    )

    return {
        "pnl_gross_d": pnl_gross_d,
        "pnl_net_d": pnl_net_d,

        "gross_PnL": pnl_gross_d.sum(),
        "net_PnL": pnl_net_d.sum(),

        "gross_SR": mySR(pnl_gross_d, scale=scale),
        "net_SR": mySR(pnl_net_d, scale=scale),

        "gross_CR": pnl_gross_d.mean() / pnl_gross_d.std(),
        "net_CR": pnl_net_d.mean() / pnl_net_d.std(),

        "av_daily_ntrans": ntrans_d.mean(),
    }



#FINAL SUMMARY OF DEVELOPPED MODEL
pnl_net_store_NQ = {}
pnl_net_store_SP = {}
summary_data1_all_quarters = pd.DataFrame()

for quarter in quarters:

    data1 = pd.read_parquet(f"input/data1_{quarter}.parquet")
    data1.set_index("datetime", inplace=True)

    # ---------------- Assumption 1 ----------------
    data1.loc[data1.between_time("9:31", "9:40").index] = np.nan
    data1.loc[data1.between_time("15:51", "16:00").index] = np.nan

    # ---------------- Assumption 2 ----------------
    pos_flat = np.zeros(len(data1))
    pos_flat[data1.index.time <= pd.to_datetime("9:55").time()] = 1
    pos_flat[data1.index.time >= pd.to_datetime("15:40").time()] = 1

    # ==================================================
    # =================== NASDAQ =======================
    # ==================================================
    signalEMA_NQ = data1["NQ"].ewm(span=10).mean()
    slowEMA_NQ   = data1["NQ"].ewm(span=120).mean()
    vol_NQ       = data1["NQ"].rolling(120).std()

    pos_NQ = volatility_breakout_position(
        signalEMA_NQ, slowEMA_NQ, vol_NQ, m=2, pos_flat=pos_flat
    )

    price_diff = data1["NQ"].diff()
    pnl_gross  = np.nan_to_num(pos_NQ * price_diff * 20)
    ntrans     = np.abs(np.diff(np.nan_to_num(pos_NQ), prepend=0))
    pnl_net    = pnl_gross - ntrans * 12

    m_NQ = compute_metrics(pnl_gross, pnl_net, data1["NQ"], ntrans)

    stat_NQ = (m_NQ["net_SR"] - 0.5) * np.maximum(
        0, np.log(np.abs(m_NQ["net_PnL"] / 1000))
    )

    pnl_net_store_NQ[quarter] = m_NQ["pnl_net_d"]

    plot_quarter_pnl(
        m_NQ["pnl_gross_d"],
        m_NQ["pnl_net_d"],
        quarter,
        "NQ"
    )

    # ==================================================
    # =================== S&P500 =======================
    # ==================================================
    signalEMA_SP = data1["SP"].ewm(span=20).mean()
    slowEMA_SP   = data1["SP"].ewm(span=150).mean()
    vol_SP       = data1["SP"].rolling(60).std()

    pos_SP = volatility_breakout_position(
        signalEMA_SP, slowEMA_SP, vol_SP, m=1, pos_flat=pos_flat
    )

    price_diff = data1["SP"].diff()
    pnl_gross  = np.nan_to_num(pos_SP * price_diff * 50)
    ntrans     = np.abs(np.diff(np.nan_to_num(pos_SP), prepend=0))
    pnl_net    = pnl_gross - ntrans * 12

    m_SP = compute_metrics(pnl_gross, pnl_net, data1["SP"], ntrans)

    stat_SP = (m_SP["net_SR"] - 0.5) * np.maximum(
        0, np.log(np.abs(m_SP["net_PnL"] / 1000))
    )

    pnl_net_store_SP[quarter] = m_SP["pnl_net_d"]

    plot_quarter_pnl(
        m_SP["pnl_gross_d"],
        m_SP["pnl_net_d"],
        quarter,
        "SP"
    )

    # ---------------- Summary ----------------
    summary_data1_all_quarters = pd.concat(
        [
            summary_data1_all_quarters,
            pd.DataFrame({
                "quarter": quarter,

                "gross_SR_NQ": m_NQ["gross_SR"],
                "net_SR_NQ":   m_NQ["net_SR"],
                "gross_PnL_NQ": m_NQ["gross_PnL"],
                "net_PnL_NQ":   m_NQ["net_PnL"],
                "gross_CR_NQ":  m_NQ["gross_CR"],
                "net_CR_NQ":    m_NQ["net_CR"],
                "av_daily_ntrans_NQ": m_NQ["av_daily_ntrans"],
                "stat_NQ": stat_NQ,

                "gross_SR_SP": m_SP["gross_SR"],
                "net_SR_SP":   m_SP["net_SR"],
                "gross_PnL_SP": m_SP["gross_PnL"],
                "net_PnL_SP":   m_SP["net_PnL"],
                "gross_CR_SP":  m_SP["gross_CR"],
                "net_CR_SP":    m_SP["net_CR"],
                "av_daily_ntrans_SP": m_SP["av_daily_ntrans"],
                "stat_SP": stat_SP,
            }, index=[0])
        ],
        ignore_index=True
    )

# =================== FULL SAMPLE PLOTS ===================
plot_total_net_pnl(pnl_net_store_NQ, "NQ")
plot_total_net_pnl(pnl_net_store_SP, "SP")


summary_data1_all_quarters

#Cumulative PnL value
total_net_pnl_nq = summary_data1_all_quarters['net_PnL_NQ'].sum()
total_net_pnl_nq

#Cumulative PnL value
total_net_pnl_sp = summary_data1_all_quarters['net_PnL_SP'].sum()
total_net_pnl_sp 

#Evaluation by defined 'stat' metric
summary_data1_all_quarters.sort_values(by = 'stat_NQ', 
                                 ascending = False)

#Statistical analysis
summary_data1_all_quarters.describe()

stat_NQ = summary_data1_all_quarters['stat_NQ'].mean() # mean value
stat_NQ

net_SR_NQ = summary_data1_all_quarters['net_SR_NQ'].mean() # mean value
net_SR_NQ

net_CR_NQ = summary_data1_all_quarters['net_CR_NQ'].mean() # mean value
net_CR_NQ

#Saving final output
summary_data1_all_quarters.to_csv("summary_data1_all_quarters.csv", index=False)