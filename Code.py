# Подключаем библиотеки

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, metrics

# Визуализация дерева

%matplotlib inline 
from IPython.display import display, HTML, SVG

style = "<style>svg{width:100% !important;height:60% !important;}</style>"
HTML(style)

# Подключаем данные

df = pd.read_csv("CleanedData.csv")

# Проверка 

df.head()

# Меняем колонку "Sex" под цифровой вариант 1 или 0

Sex = {'female':0,'male':1}
df.head()

#Удаляем целевой столбец:

X = df.drop(['Survived'], axis=1)
y = df['Survived']

# Разбиваем выборку на тренировочную и тестовую:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

# Создание дерева

clf = tree.DecisionTreeClassifier()

# Диапазон параметров для оптимального размера и метода дерева

prmtr = {'min_samples_leaf':np.arange(20,50,5),
              'min_samples_split':np.arange(20,50,5),
              'max_depth':np.arange(3,6),
              'min_weight_fraction_leaf':np.arange(0,0.4,0.1),
              'criterion':['gini','entropy']}

grid_clf = GridSearchCV(clf, prmtr, cv = 5)

# Обучение дерева

grid_clf.fit(X_train, y_train)

# Подбор лучших параметров и сохраение в переменную "best_params"

best_params = grid_clf.best_params_

y_pred = grid_clf.predict(X_test)

# Проверка точности

accuracy_score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Создание дерева с оптимальными параметрами

clf_tuned = tree.DecisionTreeClassifier(**best_params)
clf_tuned.fit(X_test, y_test)

# Визуализация дерева

import graphviz 

graph = tree.export_graphviz(clf_tuned, 
                                out_file=None,
                                filled=True, 
                                rounded=True,  
                                special_characters=True,
                               feature_names = X.columns,
                            class_names=['Dead', 'Survived']) 
graph = graphviz.Source(graph)
graph


