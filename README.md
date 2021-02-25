# Data science | Project: "Titanic"

## Гайд по проекту

Перед началом:
- "README.txt" - Содержит сам проект с визуализациями. Предпологаеться что читатель и будет смотреть его как основной файл на входе.
- "DataFrames" - Содержит в себе все дата сеты.
- "Images" - Содержит в себе все картинки "README.txt".
- "Code_Visual" - Содержит в себе код для всего визуала (Графиков и 3D).
- "Code_Tree" - Содержит в себе сам алгоритм дерева.

p.s. Во всём коде дополнительно были сделани заметки для удобства чтения и понимания.

## Разделы

- Введение
- Задача
- Exploratory Data Analysis (Анализ данных, более подробно можно просмотреть в Code_Visuals.py)
  - Чистка данных 
  - Поиск корреляций
  - Визуализация
  - Выбор параметров для обучения
- Classification Tree (Алгоритм распределения, более подробно можно просмотреть в Code_Tree.py)
  - Чистка и подготовка данных под обучение
  - Первичное обучение
  - Подбор оптимальных параметров дерева
  - Тестирование
- Итог

## Введение

Крушение Титаника - одно из самых печально известных кораблекрушений в истории.

15 апреля 1912 года, во время своего первого рейса, широко известный «непотопляемый» Титаник затонул после столкновения с айсбергом. К сожалению, на борту не хватило спасательных шлюпок на всех, в результате чего погибли 1502 человека из 2224 пассажиров и членов экипажа.

Хотя в выживании был определенный элемент удачи, похоже, некоторые группы людей выживали с большей вероятностью, чем другие.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Titanic_picture.jpg)

## Задача

В этой задаче мы  построим прогностическую модель, которая отвечает на вопрос: «Какие типы людей выживут с большей вероятностью?» с использованием данных о пассажирах (например, имя, возраст, пол, социально-экономический класс и т. д.). А также проанализируем какие факторы влияли на выживаемость и в какой степени.

Для этого предоставлено 3 дата сета (которые вы можете просмотреть в папке "DataFrames"):
- "gender_submission.csv"
- "train.csv"
- "test.csv"

p.s. Во время работы над проектом я дополнительно создавал новые "чистые" сеты (Без всего что было для меня лишним) которые я не загружал на Git, так как по сути они были сделаны для моего личного удобства, так что их можно увидеть в некоторых частях кода.


## EDA (Exploratory Data Analysis)

После чистки данных нужно было выделить основные зависимости и провести небольшой анализ взаимосвязей. Также нужно быть уверенным в том что они имеют смысл. Для начала анализа мною был выбран график для визуализации корреляций.

### Корреляции

На этом графике сразу стоит отметить несколько интересных нам корреляций.
- "Sex" (Пол)
- "Pclass" (Класс) 
- "Age" (Возраст)
Не обращаем внимания на "SibSp" и "Parch" (это логическая корреляция членов одной семьи) в данном случае нас больше всего интересует колонка "Survived".

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Корреляции.png)


### Визуализация выживших по полу

Тут мы проверяем первую основную взаимосвязь. На данном графике мы видим что вероятность выживания приблизительно в 2 раза выше если пол женский.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выжившие_по_полу.png)


### Процент выживших по классу

На этом графике мы взяли процентное соотношение выживаемости чтобы компенсировать разницу в количестве людей в разных классах. Тут мы отмечаем что 1й класс имел крайне высокую вероятность выживания, по сравнению во вторым и 3м. Предположительно это могло быть связанно с более высоким расположением коют.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выживших_по_классу.png)


### Визуализация выживших по возрасту 

Анализ кореляции возраста у меня вызвал неоднозначные мысли, так что было принято решение углубиться с помощью 3Д визуализации.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выжившие_по_возрасту.png)


### Общая 3D визуализация

Рассмотрев данную визуализацию мною было предположено что возраст хоть и влял на выживемость, так как дети в основном выживали, но при этом нужно было и учитывать что на корабле стариков практически небыло. Но у меня всё еще была догадка что тут была взаимосвязь с классом которую стоили проверить. Так что я продолжил.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Visual_3D.png)

### Выжившие по классу в соотношении с возрастом

И тут я сделал графики по возрасту по каждому классу, а также их выживаемость. Что в итоге и следовало ожидать, общая тенденция показывала смещение в правую сторону по возрасту (тоесть чем выше класс, тем старше). Что и имело такой эффект на корреляцию  возраст к выживаемости, что по сути логично. Люди более старшего возраста в основоном имеют больший достаток что в итоге позволило брать более высокий класс. Как итог нужно быть осторожним с возрастом при настройке алгоритма. 

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Выжившие_по_классу_в_соотношении_с_возрастом.png)

(Для составления всех графиков по возрасту не использовались колонки в которых NA значении в графе "Age")

## Classification Tree

У меня было 2 выбора, создавать единое дерево или лес. Лес предположительно давал бы более точную классификацию и вероятно был бы более правильным решением. Но в данном случае целью моего личного интереса было создать единое древо классификации и сделать его максимально точнным.

### Чистка и подготовка данных под обучение

Перед началом задания нам предоставили 3 дата сета (Можно просмотреть в папке "DataFrames").
- "gender_submission.csv" - Сет с информацией по Id и выживанию
- "train.csv" - Сет на котором мы будем тренеровать и проверять наш алгоритм
- "test.csv" - Цель нашей задачи (Предсказать эти данные)

Сразу же соединяем "gender_submission.csv" с "train.csv" по ключу "PassengerId" и убираем все лишние колонки кроме тех которые мы отметили для себя в EDA (Те который имеют корреляцию с выживаемостью).

### Первичное обучение



### Подбор оптимальных параметров дерева

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Tree_Accuracy.png)

Основываясь на этом графике подгоняем оптимальные парамерты глубины и тестируем. Получаем результат точности ≈ 0.83 (Округлил по правилам) соответственно 83%.

### Тестирование

Проводим Cross Value Score из 5ти. Для большей статистики и понимания тенденции во избежание ошибок.

Получаем: 0.77094972, 0.78089888, 0.84831461, 0.79213483, 0.81460674

Средняя точность: 0.8014 со среднеквадрическим отклонением (+/- 0.0552) 

Основываясь на модели классификации дерева решений, я предсказал, выживут ли пассажиры с вероятностью в ≈80% .

## Итог

Я добавил ещё пару строк для визуализации самого дерева, чтобы всё было понятней и вот что получилось. Процент предсказания довольно не плохой, учитывая что это дерево можно использовать для анализа любых случайно взятых данных из титаника. Я старался найти оптимальный баланс между его размерами чтобы дерево не было переобученым.

![alt text](https://github.com/Aettio/DS_Project_Titanic/blob/main/Images/Classification_Tree.jpg)

(Полный код вы можете просмотерть в Code_Tree.py)
