import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree

sns.set()

# Import data
df_train = pd.read_csv('Train_dataset.csv')
df_test = pd.read_csv('Test_dataset.csv')
# Store target variable of training data in a safe place
infect_train = df_train.Infect_Prob

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Infect_Prob'], axis=1), df_test])
data=data.dropna()


data['Diuresis'] = data.Diuresis.fillna(data.Diuresis.median())
data['Platelets'] = data.Platelets.fillna(data.Platelets.median())
data['HBB'] = data.HBB.fillna(data.HBB.median())
data['d-dimer'] = data['d-dimer'].fillna(data['d-dimer'].median())
data['salary'] = data.salary .fillna(data.salary .median())
data['Insurance'] = data.Insurance .fillna(data.Insurance .median())
data['Heart rate'] = data['Heart rate'].fillna(data['Heart rate'].median())
data['HDL cholesterol'] = data['HDL cholesterol'].fillna(data['HDL cholesterol'].median())
data['Charlson Index'] = data['Charlson Index'].fillna(data['Charlson Index'].median())
data['Blood Glucose'] = data['Blood Glucose'].fillna(data['Blood Glucose'].median())
data['FT/month'] = data['FT/month'] .fillna(data['FT/month'].median())
data['Coma score'] = data['Coma score'] .fillna(data['Coma score'].median())
data['Age'] = data['Age'] .fillna(data['Age'].median())
data['Deaths/1M'] = data['Deaths/1M'] .fillna(data['Deaths/1M'].median())
data['cases/1M'] = data['cases/1M'] .fillna(data['cases/1M'].median())

data = data[['Age','Coma score','cases/1M','Deaths/1M','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','Insurance','salary','FT/month']]
data=data.dropna()

# print(data.info())

data_train = data.iloc[:891]
data_test = data.iloc[891:]
data_train=data_train.dropna()
data_test=data_test.dropna()


new=infect_train.dropna(axis = 0, how ='any')
# print(new)

X = data_train.drop(data_train.columns[np.isnan(data_train).any()], axis=1).values
test= data_test.drop(data_test.columns[np.isnan(data_test).any()], axis=1).values
y = new.values

clf=tree.DecisionTreeRegressor()
# clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y[:891])

Y_pred = clf.predict(test)

df_test['Infect_Prob'] = Y_pred[:14498]

df_test[['people_ID','Infect_Prob']].to_csv('part1.csv', index=False)
print('Finish')