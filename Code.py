# Подключаем библиотеки

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import tree, metrics

# Подключаем данные

df = pd.read_csv("CleanedData.csv")

#Удаляем целевой столбец:

X = df.drop(['Survived'], axis=1)
y = df['Survived']

# Разбиваем выборку на тренировочную и тестовую:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
