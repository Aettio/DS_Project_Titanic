# Подключаем библиотеки

import pandas as pd 
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score

# Открываем данные

train_data = pd.read_csv("C:/Users/baran/Desktop/Datasets/Titanic/train.csv")
gs = pd.read_csv("C:/Users/baran/Desktop/Datasets/Titanic/gender_submission.csv")
test_data = pd.read_csv("C:/Users/baran/Desktop/Datasets/Titanic/test.csv")

# Чистка данных train_data

train_data = train_data.drop(['Name','Ticket', "Embarked","Cabin"], axis=1)

clean_train = pd.merge(train_data, gs, left_on='PassengerId', right_on='PassengerId', how='left')\
    .drop('Survived_y', axis=1)\
    .rename(columns={'Survived_x': 'Survived'})
clean_train = clean_train.drop(['PassengerId'], axis=1)

# Чистка данных test_data

clean_test = test_data.drop(["PassengerId", 'Name','Ticket', "Embarked","Cabin"], axis=1)

# Меняем колонку "Sex" под цифровой вариант male = 1 и female = 0

clean_test['Sex'] = clean_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
clean_train['Sex'] = clean_train['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Функция разделения возраста по группам и заполнение всех NA (Включая Fare) медианой.

def simplify_age(clean_train):
    clean_train['Age'].fillna(clean_train['Age'].median(), inplace=True)
    clean_train['Fare'].fillna(clean_train['Fare'].median(), inplace=True)
    
    clean_train.loc[clean_train['Age'] <= 16, 'Age'] = 0
    clean_train.loc[(clean_train['Age'] > 16) & (clean_train['Age'] <= 32), 'Age'] = 1
    clean_train.loc[(clean_train['Age'] > 32) & (clean_train['Age'] <= 48), 'Age'] = 2
    clean_train.loc[(clean_train['Age'] > 48) & (clean_train['Age'] <= 64), 'Age'] = 3
    clean_train.loc[clean_train['Age'] > 48, 'Age'] = 4   
    return clean_train

clean_train = simplify_age(clean_train)

# Удаляем целевой столбец:

X = clean_train.drop(["Survived"], axis=1)
y = clean_train['Survived']

# Разбиваем выборку на тренировочную и тестовую:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Создаём массивы для хранения данных для графика 

dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Создание дерева первой версии дерева

for i, k in enumerate(dep):
  
    # Создание дерева
    clf = tree.DecisionTreeClassifier(max_depth=k)
    clf.fit(X_train, y_train)

    # Запись данных для графика точности (Train)
    train_accuracy[i] = clf.score(X_train, y_train)

    # Запись данных для графика точности (Test)
    test_accuracy[i] = clf.score(X_test, y_test)
    
# Диапазон параметров для оптимального размера и метода дерева

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set()

# Создание графика

plt.title('clf: Соотношение глубины к результатам предсказания')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()

# Улучшение дерева 

clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=1)
clf.fit(X, y)

# Проверка точности 

clf.score(X, y)

# Проверка точности Cross Value Score (5 Раз)

from sklearn.model_selection import train_test_split, cross_val_score

cvs = cross_val_score(clf,X,y,cv=5)
print(cvs)

# Средне арефмитическое Cross Value Score со среднеквадрическим отклонением 

print("Accuracy: %0.4f (+/- %0.4f)" % (cvs.mean(), cvs.std()*2))

# Визуализация дерева

import graphviz 

graph = tree.export_graphviz(clf, 
                                out_file=None,
                                filled=True, 
                                rounded=True,  
                                special_characters=True,
                               feature_names = X.columns,
                            class_names=['Dead', 'Survived']) 
graph = graphviz.Source(graph)
graph

