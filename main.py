# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:39:10 2023

@author: harrisfs
"""
import seaborn as sns
import xgboost as  xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()

cwd = os.getcwd()

df = pd.read_csv(cwd + '/data/energy_data.csv')
df = df.set_index('datetime')

# check consumption over time for exploration
# df.plot(style='.', 
#         figsize=(15,5),
#         color=color_pal[0],
#         title = "Energy use (MW)")

# df.index = pd.to_datetime(df.index)

# energy consumption histogram
df['energy_MW'].plot(kind='hist', bins=30, title='Histogram of Energy Consumption', figsize=(10, 6))
plt.xlabel('Energy (MW)')
plt.show()


#cumulative distribution graph
sorted_data = np.sort(df['energy_MW'])
cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

plt.figure(figsize=(10, 6))
plt.plot(sorted_data, cdf, marker='.', linestyle='none')
plt.xlabel('Energy (MW)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Probability Distribution of Energy Consumption')
plt.grid(True)
plt.show()


# random week consumption
df.loc[(df.index >= '01-01-2010') & (df.index < '01-08-2010')].plot()

# simple data cleaning due to visible outlier
df = df.query('energy_MW >=19000').copy()

# feature creation
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)


# monthly boxplot
df['month'] = df.index.month

fig, ax = plt.subplots(figsize=(10, 6))
boxprops = dict(facecolor='lightblue', color='black')
medianprops = dict(color='red', linewidth=2)

df.boxplot(column='energy_MW', by='month', ax=ax, boxprops=boxprops, medianprops=medianprops, patch_artist=True)

plt.title('Monthly Energy Consumption')
plt.suptitle('')
plt.xlabel('Month')
plt.ylabel('Energy (MW)')
plt.show()



# dataset split
train = df.loc[df.index < '01-01-2016']
test = df.loc[df.index >= '01-01-2016']


features = ['hour', 'dayofweek', 'quarter', 'month', 'year','dayofyear']
target = 'energy_MW'

# model creation
x_train = train[features]
y_train = train[target]
x_test = test[features]
y_test = test[target]

reg = xgb.XGBRegressor(n_estimators=1000,early_stopping_rounds = 50,learning_rate = 0.01)
reg.fit(x_train,y_train, 
        eval_set=[(x_train,y_train),(x_test,y_test)],
        verbose = 100)

# feature importance
f1 = pd.DataFrame(data=reg.feature_importances_,
                  index=reg.feature_names_in_,
                  columns = ['importance'])

# forecast on test set
test['prediction'] = reg.predict(x_test)
df = df.merge(test[['prediction']], how='left', left_index=True,right_index=True)

ax = df[['energy_MW']].plot(figsize=(15,5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['true data','predictions'])
ax.set_title('Raw Data and Predictions')
plt.show()

np.sqrt(mean_squared_error(test['energy_MW'], test['prediction']))

test['error'] = np.abs(test[target] - test['prediction'])

test['date'] = test.index.date

test.groupby('date')['error'].mean().sort_values(ascending = False)

# save model to avoid re-training in future scripts
reg.save_model('model.json')
