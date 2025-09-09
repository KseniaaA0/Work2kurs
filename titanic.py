import pandas as pd
import numpy as ny
from sklearn.preprocessing import OneHotEncoder

from sklearn. model_selection import train_test_split
from sklearn. linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

test = pd.read_csv('/content/test.csv')
ds = pd.read_csv('/content/train.csv')


from google.colab import drive
drive.mount('/content/drive')

ds.shape

ds.isnull().mean() * 100

ds = ds.drop(columns=['Cabin'])
ds = ds.drop(columns=['Name'])
ds = ds.drop(columns=['PassengerId'])
ds = ds.drop(columns=['Ticket'])

ds['Sex'].replace(['male', 'female'],
                        [0, 1], inplace=True)


ds = pd.get_dummies(ds, columns = ['Embarked'])

df = ds.boxplot(column=['Age'], figsize=(2, 7))
df

ds.describe()

print(ds.corr())


ds.groupby(by='Pclass').mean()

ds['Age'].fillna(ds['Age'].mean(), inplace=True)
ds

ds = ds.dropna()
x = ds.drop(columns=['Age','Embarked_C','Embarked_Q','Embarked_S',])
y = ds['Age']
lreg = LinearRegression().fit(x,y)
lreg.score(x,y) 


model = OLS(y, x)
res = model.fit()
print(res.summary())

ds

X = ds.drop(['Survived'], axis=1) 
Y = ds['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

ds

log_reg_survival = LogisticRegression()
log_reg_survival.fit(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier
knn_embarked = KNeighborsClassifier()
knn_embarked.fit(X_train, Y_train)

survival_predictions = log_reg_survival.predict(X_test)
survival_accuracy = accuracy_score(Y_test, survival_predictions)

embarked_predictions = knn_embarked.predict(X_test)
print("Точность логистической регрессии выживания:", survival_accuracy)
embarked_accuracy = accuracy_score(Y_test, embarked_predictions)
print("Точность модели kNN эмбаркации:", embarked_accuracy)
