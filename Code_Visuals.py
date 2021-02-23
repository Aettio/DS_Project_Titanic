## Подключаем библиотеки

import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize":(10,8)})

sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

# Открываем данные

train_data = pd.read_csv("C:/Users/baran/Desktop/Datasets/Titanic/train.csv")
gs = pd.read_csv("C:/Users/baran/Desktop/Datasets/Titanic/gender_submission.csv")

# Чистка данных train_data

train_data = train_data.drop(['Ticket', "Fare", "Embarked","Cabin"], axis=1)

clean_train = pd.merge(train_data, gs, left_on='PassengerId', right_on='PassengerId', how='left')\
    .drop('Survived_y', axis=1)\
    .rename(columns={'Survived_x': 'Survived'})

# Данные для 3D визуала
graph_3d = train_data

## EDA
# Поиск самых значмых корреляций

clean_train = clean_train.drop(['PassengerId'], axis=1)
sns.heatmap(clean_train.corr(), annot = True, cmap= 'viridis', linewidths=0.2, linecolor="black")

# Визуализация выживших по полу

sns.histplot(data = clean_train, x = "Sex", hue = "Survived", multiple="stack", shrink= 0.9, palette="viridis")

# Визуализация выживших по возрасту 

sns.histplot(data = clean_train, x = "Age", hue = "Survived", multiple="stack", palette="viridis")

# 3D Визуал

import plotly.express as px

fig = px.scatter_3d(graph_3d, x = "Name", y='Sex', z='Age', color = 'Survived')
fig.show()

# Процент выживших по классу

sns.barplot(data=clean_train, x = "Pclass",y = "Survived", palette="viridis")

# Выжившие по классу в соотношении с возрастом

grid = sns.FacetGrid(clean_train, col='Survived',hue = "Survived", row='Pclass', height=4, aspect=1.5, palette="viridis")


grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend()
