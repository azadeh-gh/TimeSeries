#import packages
from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.plotting import autocorrelation_plot

# Import data as Dataframe
df = pd.read_csv('../PPT_data/PPT_data.csv')
df["Date"]=pd.to_datetime(df["Date"], format="%m/%d/%Y")

#define a function to remove outliers
def remove_outlier(DF,col_name,coef):
    originalArray=DF[col_name]
    Mean=np.mean(originalArray)
    std=np.std(originalArray)
    bool=((DF[col_name]>Mean+coef*std) | (DF[col_name]<Mean-coef*std))
    CleanedDF=DF[~bool]
    return CleanedDF

#use function to remove outliers
df=remove_outlier(df,"PPT[mm]",3)

#Define start and end date that will use
startDate=datetime.datetime.strptime("01/01/2017", "%m/%d/%Y")
EndDate = datetime.datetime.strptime("12/30/2019", "%m/%d/%Y")

#Define Boolean and select dates between start date and end date
BoolL_Data = ((df["Date"] > startDate) & (df["Date"] < EndDate))
df = df[BoolL_Data]

#Define a function and use matplotlib to visualise the series
def plot_df( x, y, title="", xlabel='Date', ylabel='PPT', dpi=100):
    plt.figure(figsize=(10,4), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.savefig('../Results/PPT_timeSeries.png')
    plt.close()


# Use The defined function to visualize a PPT time series
plot_df(x=df['Date'], y=df["PPT[mm]"], title='Daily precepitation [mm].')

#Define a boolean to remove zero values
bool=df["PPT[mm]"]==0
df=df[~bool]

#Boxplot of Month-wise (Seasonal) and Year-wise (trend) Distribution

# Prepare data
df['year'] = [d.year for d in df["Date"]]
df['month'] = [d.strftime('%b') for d in df["Date"]]
years = df['year'].unique()

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(10,4), dpi= 80)
sns.boxplot(x='year', y='PPT[mm]', data=df, ax=axes[0])
sns.boxplot(x='month', y='PPT[mm]', data=df.loc[~df.year.isin([2017, 2020]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18);
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)


#To change the label sizes
axes[0].set_ylabel("PPT [mm]", fontsize=14)
axes[0].set_xlabel("Year", fontsize=14)
axes[1].set_ylabel("PPT [mm]", fontsize=14)
axes[1].set_xlabel("Month", fontsize=14)
plt.savefig('../Results/PPT_Boxplots.png')
plt.close()

#Patterns in time series

# Decompose PPT time series into its component

#Make "Date" column as index
df.set_index("Date", inplace=True)

# Multiplicative Decomposition
result_mul = seasonal_decompose(df['PPT[mm]'], model='multiplicative', extrapolate_trend='freq',period=3)

# Additive Decomposition
result_add = seasonal_decompose(df['PPT[mm]'], model='additive', extrapolate_trend='freq',period=3)

# Plot
plt.rcParams.update({'figure.figsize': (10,12)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.close()


#Put additive decomposition results in a DataFrame
df_reconstructed_add = pd.concat([result_add.seasonal, result_add.trend, result_add.resid, result_add.observed],axis=1)

#Define column names for DataFrame
df_reconstructed_add.columns = ['seas', 'trend', 'resid', 'actual_values']

#Another way to Plot decomposition data with more setting
def plot_decompose(DF,colName,ylabel):
    plt.gca()
    plt.plot(DF[colName], color="blue")
    plt.title("Additive decomposition", fontsize=16)
    plt.ylabel(ylabel, fontsize=14)


plt.figure(figsize=(10,10))
plt.subplot(4,1,1)
plot_decompose(df_reconstructed_add,"actual_values","Observed")

plt.subplot(4, 1, 2)
plot_decompose(df_reconstructed_add,"trend","Trend")

plt.subplot(4, 1, 3)
plot_decompose(df_reconstructed_add,"seas","Seasonal")

plt.subplot(4, 1, 4)
plt.gca()
plt.scatter(df_reconstructed_add.index,df_reconstructed_add["resid"],color="blue",facecolor="none")
plt.axhline(y=0)
plt.ylabel("Residual",fontsize=14)
plt.xlabel("Date", fontsize=14)
plt.tight_layout()
plt.savefig('../Results/Decompose.png')
plt.close()

# Stationarity test using Augmented Dickey Fuller test (ADH Test) and  Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary)


# ADF Test
#the null hypothesis is the time series possesses a unit root and is non-stationary. So, id the P-Value in ADH test is less than the significance level (0.05), you reject the null hypothesis.
result = adfuller(df["PPT[mm]"].values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
#The null hypothesis and the P-Value interpretation is just the opposite of ADH test. The below code implements these two tests using statsmodels package in python.
result = kpss(df["PPT[mm]"].values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

#seasonality test of a time series

# Draw Plot
plt.rcParams.update({'figure.figsize':(12,2), 'figure.dpi':120})
autocorrelation_plot(df["PPT[mm]"].tolist())
plt.savefig('../Results/autocorrelation.png')